# Lets GitHub Actions assume an AWS role via OIDC federation instead of storing long-lived AWS
# access keys as GitHub secrets. Scoped to this repo, and (for deploy-level permissions) to the
# main branch only - PR/build-only CI jobs need no AWS access at all.

data "tls_certificate" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # AWS STS validates the token against the provider's published JWKS rather than this
  # thumbprint in practice, but the API still requires one - fetched from the live cert rather
  # than hardcoded, so it's correct regardless of when GitHub rotates their CA.
  thumbprint_list = [data.tls_certificate.github_actions.certificates[0].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "${var.project_name}-github-actions-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
}

data "aws_iam_policy_document" "github_actions_deploy" {
  # `terraform plan`/`apply` refreshes every resource already in state, not just the one being
  # changed (the Lambda's image_uri) - and the AWS provider's refresh of a resource like
  # aws_s3_bucket calls a long tail of sub-config read APIs (CORS, logging, lifecycle, website,
  # replication, ...) that aren't obvious from the resource's own .tf block. Enumerating exact
  # read-only actions one at a time turned into repeated whack-a-mole (missing one blocks the
  # whole apply, and any manual console fix gets overwritten by the next apply anyway, since
  # Terraform reconciles this policy to exactly match this file). So: every *write* action below
  # stays explicitly enumerated and narrow - that's what actually matters for blast radius - but
  # read-only verbs (Get/List/Describe) are wildcarded, each still scoped to one specific
  # resource ARN this project owns, not to the account.

  statement {
    sid       = "STSIdentity"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }

  statement {
    sid       = "ECRAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "ECRRepoRead"
    actions = [
      "ecr:Describe*",
      "ecr:Get*",
      "ecr:BatchGet*",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = ["arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${var.ecr_repository_name}"]
  }

  statement {
    sid = "ECRRepoPush"
    actions = [
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = ["arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${var.ecr_repository_name}"]
  }

  statement {
    sid     = "TerraformStateObjects"
    actions = ["s3:Get*", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.tfstate.arn,
      "${aws_s3_bucket.tfstate.arn}/*",
    ]
  }

  statement {
    # Covers every S3 bucket-level config read (CORS, website, logging, lifecycle, replication,
    # ACL, policy, versioning, encryption, ...) the aws_s3_bucket resource's refresh might call -
    # still scoped to just this one state bucket, no object-level reach (that's the statement
    # above). s3:List* is for the bucket itself (e.g. ListBucketVersions), not object listing.
    sid       = "TerraformStateBucketConfig"
    actions   = ["s3:Get*", "s3:List*"]
    resources = [aws_s3_bucket.tfstate.arn]
  }

  statement {
    sid     = "LambdaRead"
    actions = ["lambda:Get*", "lambda:List*"]
    resources = [
      aws_lambda_function.api.arn,
      "${aws_lambda_function.api.arn}:*",
    ]
  }

  statement {
    sid = "LambdaManage"
    actions = [
      "lambda:UpdateFunctionCode",
      "lambda:UpdateFunctionConfiguration",
      "lambda:TagResource",
      # Direct synchronous invoke for the post-deploy warm-up call - bypasses API Gateway's
      # hard 30s integration timeout cap, so a worst-case cold start has real room to finish.
      "lambda:InvokeFunction",
    ]
    resources = [aws_lambda_function.api.arn]
  }

  statement {
    sid       = "ApiGatewayRead"
    actions   = ["apigateway:GET"]
    resources = ["arn:aws:apigateway:${var.aws_region}::/apis/*"]
  }

  statement {
    # CloudWatch Logs' Describe/List actions don't support resource-level restriction, so this
    # is necessarily account-wide - but it's read-only (no Put/Delete) and scoped to this one
    # service, not a broader wildcard.
    sid       = "LogsRead"
    actions   = ["logs:Describe*", "logs:List*"]
    resources = ["*"]
  }

  statement {
    sid     = "IAMRoleRead"
    actions = ["iam:Get*", "iam:List*"]
    resources = [
      aws_iam_role.lambda_exec.arn,
      aws_iam_role.github_actions_deploy.arn,
    ]
  }

  statement {
    sid       = "IAMOidcRead"
    actions   = ["iam:Get*", "iam:List*"]
    resources = [aws_iam_openid_connect_provider.github_actions.arn]
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${var.project_name}-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
