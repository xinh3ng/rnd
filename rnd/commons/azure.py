from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.keyvault.secrets import SecretClient

from rnd.commons.commons import create_logger

logger = create_logger(__name__, level="info")


def get_azure_secret(
    secret_name: str,
    vault_url: str,
    tenant_id: str = "your_tenant_id",
    client_id: str = "your_client_id",
    client_secret: str = "your_client_secret",
):
    """Trying the default approach. Then the client secret approach

    Example:
        xh-secrets is my kay vault. Then I must add my Azure idendity app "extraid-rnd" to its IAM

    Args:
        tenant_id: corresponds to Azure AD
        client_id: corresponds to an Azure AD application
    """
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret(secret_name)
    except Exception as e:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret(secret_name)

    return secret
