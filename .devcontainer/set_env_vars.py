import os
from dotenv import load_dotenv

# Load existing .env file if it exists
load_dotenv()

secrets = {
    # Azure Service principal (SP) credentials
    'AZURE_CLIENT_ID': os.getenv('AZURE_CLIENT_ID'),
    'AZURE_TENANT_ID': os.getenv('AZURE_TENANT_ID'),
    'AZURE_CLIENT_SECRET': os.getenv('AZURE_CLIENT_SECRET'),
    
    # Microsoft Fabric workspace and lakehouse details
    'ACCOUNT_NAME': os.getenv('ACCOUNT_NAME'),
    'WORKSPACE_ID': os.getenv('WORKSPACE_ID'),
    'WORKSPACE_NAME': os.getenv('WORKSPACE_NAME'),
    'LAKEHOUSE_ID': os.getenv('LAKEHOUSE_ID'),
    'LAKEHOUSE_NAME': os.getenv('LAKEHOUSE_NAME'),
    
    # Azure DevOps details and personal access token (PAT)
    'ADO_PERSONAL_ACCESS_TOKEN': os.getenv('ADO_PERSONAL_ACCESS_TOKEN'),
    'ADO_ORGANIZATIONAL_URL': os.getenv('ADO_ORGANIZATIONAL_URL'),
    'ADO_PROJECT_NAME': os.getenv('ADO_PROJECT_NAME'),
    'ADO_REPO_NAME': os.getenv('ADO_REPO_NAME'),
    
    # GitHub details and personal access token (PAT)
    'GH_PERSONAL_ACCESS_TOKEN': os.getenv('GH_PERSONAL_ACCESS_TOKEN'),
    'GH_USERNAME': os.getenv('GH_USERNAME'),
    'GH_REPO_NAME': os.getenv('GH_REPO_NAME'),
    'GH_TOKEN': os.getenv('GH_TOKEN'),
    'SOURCE_REPO': os.getenv('SOURCE_REPO'),
    
    # Financial API keys
    'EIA_KEY': os.getenv('EIA_KEY'),
    'FMP_API_KEY': os.getenv('FMP_API_KEY'),
    'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY'),
    'ALPHAVANTAGE_API_KEY': os.getenv('ALPHAVANTAGE_API_KEY'),
    'TWELVE_DATA_API_KEY': os.getenv('TWELVE_DATA_API_KEY'),
    'FRED_API_KEY': os.getenv('FRED_API_KEY'),
    
    # AWS credentials
    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
    
    # FINRA credentials
    'FINRA_CLIENT_ID': os.getenv('FINRA_CLIENT_ID'),
    'FINRA_CLIENT_SECRET': os.getenv('FINRA_CLIENT_SECRET'),
    
    # HuggingFace token
    'HF_TOKEN': os.getenv('HF_TOKEN')
}

# Update .env file
with open('.env', 'a') as f:
    for key, value in secrets.items():
        if value is not None:
            f.write(f"{key}={value}\n")