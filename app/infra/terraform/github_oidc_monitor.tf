# A second, separate OIDC-trusted role for the scheduled drift-monitor workflow - deliberately
# not reusing github_actions_deploy. This one only ever reads predictions / reads+writes the
# shared MLflow registry sync file; it can't push images, update the Lambda, or touch Terraform
# state. GitHub Issue/PR creation uses the workflow's own GITHUB_TOKEN, not AWS, so no IAM
# permission is needed for that part.

data "aws_iam_policy_document" "github_actions_monitor_assume_role" {
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

resource "aws_iam_role" "github_actions_monitor" {
  name               = "${var.project_name}-github-actions-monitor"
  assume_role_policy = data.aws_iam_policy_document.github_actions_monitor_assume_role.json
}

data "aws_iam_policy_document" "github_actions_monitor" {
  statement {
    sid       = "STSIdentity"
    actions   = ["sts:GetCallerIdentity"]
    resources = ["*"]
  }

  statement {
    sid       = "ReadPredictionLogs"
    actions   = ["s3:Get*", "s3:List*"]
    resources = [aws_s3_bucket.predictions.arn, "${aws_s3_bucket.predictions.arn}/*"]
  }

  statement {
    # Shared MLflow tracking/registry backend (sqlite file), synced down before training and
    # back up after promoting - see deposit_classification/modeling/README.md. Lives under its
    # own prefix in the existing state bucket rather than a new one.
    sid       = "MlflowSync"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:Get*", "s3:List*"]
    resources = [aws_s3_bucket.tfstate.arn, "${aws_s3_bucket.tfstate.arn}/mlflow/*"]
  }
}

resource "aws_iam_role_policy" "github_actions_monitor" {
  name   = "${var.project_name}-github-actions-monitor"
  role   = aws_iam_role.github_actions_monitor.id
  policy = data.aws_iam_policy_document.github_actions_monitor.json
}
