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
  # changed (the Lambda's image_uri) - so this role needs read access across everything the
  # Terraform config manages, even though only a handful of actions actually mutate anything.

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
    sid = "ECRRepo"
    actions = [
      "ecr:DescribeRepositories",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:DescribeImages",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = ["arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${var.ecr_repository_name}"]
  }

  statement {
    # s3:DeleteObject is for releasing the native S3 lock (use_lockfile) - Terraform writes a
    # `<key>.tflock` object before an operation and deletes it after, not just on state writes.
    sid     = "TerraformStateObjects"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.tfstate.arn,
      "${aws_s3_bucket.tfstate.arn}/*",
    ]
  }

  statement {
    sid = "TerraformStateBucketConfig"
    actions = [
      "s3:GetBucketVersioning",
      "s3:GetEncryptionConfiguration",
      "s3:GetBucketPublicAccessBlock",
      "s3:GetBucketAcl",
      "s3:GetBucketTagging",
      "s3:GetBucketLocation",
      "s3:GetBucketPolicy",
    ]
    resources = [aws_s3_bucket.tfstate.arn]
  }

  statement {
    sid = "LambdaManage"
    actions = [
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      "lambda:UpdateFunctionCode",
      "lambda:UpdateFunctionConfiguration",
      "lambda:ListVersionsByFunction",
      "lambda:ListTags",
      "lambda:TagResource",
      "lambda:GetPolicy",
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
    sid       = "LogsRead"
    actions   = ["logs:DescribeLogGroups", "logs:ListTagsForResource", "logs:ListTagsLogGroup"]
    resources = ["*"]
  }

  statement {
    sid = "IAMRead"
    actions = [
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListRoleTags",
    ]
    resources = [
      aws_iam_role.lambda_exec.arn,
      aws_iam_role.github_actions_deploy.arn,
    ]
  }

  statement {
    sid       = "IAMOidcRead"
    actions   = ["iam:GetOpenIDConnectProvider"]
    resources = [aws_iam_openid_connect_provider.github_actions.arn]
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "${var.project_name}-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
