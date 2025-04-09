import azure.functions as func
import logging

app = func.FunctionApp()

# Define the function with binding decorators
@app.function_name(name="ProcessTennisVideo")
@app.blob_trigger(arg_name="myblob", path="input-videos/{name}", connection="AzureWebJobsStorage")
def process_tennis_video(myblob: func.BlobClient):
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.size} bytes")
    
    # TODO: Process the video and store the result in outputBlob
    # For now, just pass through the input to output
    pass
