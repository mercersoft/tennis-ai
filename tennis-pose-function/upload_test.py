from azure.storage.blob import BlobServiceClient
import os

# Connect to the storage emulator
connection_string = "UseDevelopmentStorage=true"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create container if it doesn't exist
container_name = "input-videos"
try:
    container_client = blob_service_client.create_container(container_name)
except:
    container_client = blob_service_client.get_container_client(container_name)

# Upload the file
file_path = "../7630973683334062479.mov"
blob_name = os.path.basename(file_path)

with open(file_path, "rb") as data:
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)

print(f"Uploaded {blob_name} to {container_name}") 