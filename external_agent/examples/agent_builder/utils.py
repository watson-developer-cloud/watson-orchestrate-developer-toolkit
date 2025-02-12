import time
import json
import uuid
import os
import traceback
import logging
import requests
from typing import List
from ibm_watsonx_ai import APIClient

from models import Message


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


WATSONX_DEPLOYMENT_ID = os.getenv("WATSONX_DEPLOYMENT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_SPACE_ID = os.getenv("WATSONX_SPACE_ID")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")


def _get_access_token():
    api_key = WATSONX_API_KEY
    file_path = "./current_token.txt"
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "accept": "application/json",
    }
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}

    if os.path.isfile(file_path):
        file_time = os.path.getmtime(file_path)
        if time.time() - file_time < 3600:
            logger.info("Retrieved cached token")
            with open(file_path, "r") as file:
                return file.read()

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        token_data = json.loads(response.text)
        token = token_data["access_token"]

        with open(file_path, "w") as file:
            file.write(token)
        logger.info("Retrieved new token")
        return token
    else:
        raise Exception("Failed to get access token")


def _get_wxai_client():
    credentials = {"url": WATSONX_URL, "token": _get_access_token()}
    if WATSONX_PROJECT_ID:
            return APIClient(credentials, project_id=WATSONX_PROJECT_ID)
    if WATSONX_SPACE_ID:
        return APIClient(credentials, space_id=WATSONX_SPACE_ID)
    raise ValueError("Both WATSONX_PROJECT_ID and WATSONX_SPACE_ID are None. Please provide at least one.")

def get_llm_sync(messages: List[Message]) -> list[Message]:

    """

    wrapper around run_ai_service(sync version)

    """
    logger.info("wx.ai deployment Synchronous call")
    client = _get_wxai_client()
    payload = {"messages": [m.model_dump() for m in messages if m.role != "system"]}
    logger.info(f"Calling AI service with payload: {payload}")
    result = client.deployments.run_ai_service(WATSONX_DEPLOYMENT_ID, payload)
    if "error" in result:
        raise RuntimeError(f"Got an error from wx.ai AI service: {result['error']}")

    logger.info(f"Response: {result}")
    return [Message(**c["message"]) for c in result["choices"]]


def format_resp(struct):
    return "data: " + json.dumps(struct) + "\n\n"


async def get_llm_stream(messages: List[Message], thread_id: str):
    """

    wrapper around run_ai_service_stream(streaming version)

    note that run_ai_service_stream seems to return more messages than run_ai_service:
    - assistant messages with tool_calls instead of content
    - tool messages
    these are not returned to orchestrate
    """
    logger.info("wx.ai deployment streaming call start")
    client = _get_wxai_client()
    payload = {"messages": [m.model_dump() for m in messages if m.role != "system"]}
    logger.info(f"wx.ai deployment streaming call payload {payload}")
    try:
        for chunk in client.deployments.run_ai_service_stream(
            WATSONX_DEPLOYMENT_ID, payload
        ):
            logger.info(f"Received chunk from AI service: {chunk}")
            result = json.loads(chunk)["choices"][0]["message"]

            if result["role"] != "assistant" or "delta" not in result:
                continue

            current_timestamp = int(time.time())
            struct = {
                "id": str(uuid.uuid4()),
                "object": "thread.message.delta",
                "created": current_timestamp,
                "thread_id": thread_id,
                "model": "wx.ai AI service",
                "choices": [
                    {
                        "delta": {
                            "content": result["delta"],
                            "role": "assistant",
                        }
                    }
                ],
            }
            event_content = format_resp(struct)
            logger.info("Sending event content: " + event_content)
            yield event_content
    except Exception as e:
        logger.error(f"Exception {str(e)}")
        traceback.print_exc()
        yield f"Error: {str(e)}\n"
