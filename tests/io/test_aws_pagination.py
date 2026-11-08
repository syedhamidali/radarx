"""Exercise S3 continuation using the real botocore paginator, without AWS."""

import boto3
import pytest
from botocore.stub import Stubber

from radarx.io import aws_data


@pytest.fixture
def client(monkeypatch):
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    calls = []

    def get_client(anonymous=True):
        calls.append(anonymous)
        return client

    monkeypatch.setattr(aws_data, "get_s3_client", get_client)
    with Stubber(client) as stub:
        yield stub, calls


def params(token=None):
    result = {"Bucket": "example-bucket", "Prefix": "radar/"}
    if token is not None:
        result["ContinuationToken"] = token
    return result


@pytest.mark.parametrize("anonymous", [True, False])
def test_multiple_pages_preserve_service_order_and_access(client, anonymous):
    stub, calls = client
    keys = [f"radar/{i:04}" for i in range(1000)]
    stub.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": key} for key in keys],
            "IsTruncated": True,
            "NextContinuationToken": "opaque-next",
        },
        params(),
    )
    stub.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": "radar/final"}, {"Key": keys[0]}],
            "IsTruncated": False,
        },
        params("opaque-next"),
    )
    assert aws_data.list_available_files(
        "example-bucket", "radar/", anonymous
    ) == keys + ["radar/final", keys[0]]
    stub.assert_no_pending_responses()
    assert calls == [anonymous]


@pytest.mark.parametrize(
    "page", [{}, {"Contents": []}, {"Contents": [{"Key": "radar/z"}]}]
)
def test_single_page_and_empty_prefix(client, page):
    stub, calls = client
    stub.add_response("list_objects_v2", page, params())
    assert aws_data.list_available_files("example-bucket", "radar/") == [
        x["Key"] for x in page.get("Contents", [])
    ]
    assert calls == [True]
    stub.assert_no_pending_responses()


def test_empty_intermediate_page_does_not_stop_listing(client):
    stub, _ = client
    stub.add_response(
        "list_objects_v2",
        {"IsTruncated": True, "NextContinuationToken": "next"},
        params(),
    )
    stub.add_response(
        "list_objects_v2", {"Contents": [{"Key": "radar/last"}]}, params("next")
    )
    assert aws_data.list_available_files("example-bucket", "radar/") == ["radar/last"]
    stub.assert_no_pending_responses()


@pytest.mark.parametrize("later", [False, True])
def test_client_error_returns_no_partial_result(client, capsys, later):
    stub, _ = client
    if later:
        stub.add_response(
            "list_objects_v2",
            {
                "Contents": [{"Key": "radar/first"}],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            },
            params(),
        )
    stub.add_client_error(
        "list_objects_v2",
        service_error_code="AccessDenied",
        service_message="denied",
        expected_params=params("next" if later else None),
    )
    assert aws_data.list_available_files("example-bucket", "radar/") == []
    assert "Failed to list files:" in capsys.readouterr().out
    stub.assert_no_pending_responses()
