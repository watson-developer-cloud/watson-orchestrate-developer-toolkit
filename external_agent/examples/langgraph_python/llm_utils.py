import time
import json
import uuid
import traceback
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage, ToolCall
from langgraph.prebuilt import create_react_agent
from ibm_watsonx_ai import APIClient, Credentials
from langchain_ibm import ChatWatsonx
from models import Message, AIToolCall, Function, ChatCompletionResponse, Choice, MessageResponse
from config import OPENAI_API_KEY, WATSONX_SPACE_ID, WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID
from token_utils import get_access_token

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_openai(model: str, parm_overrides: dict = {}):
    defaults = {
        'temperature': 0,
        'streaming': False
    }
    defaults.update(parm_overrides)
    return ChatOpenAI(model=model, **defaults)

def convert_messages_to_langgraph_format(messages: List[Message]) -> Dict[str, Any]:
    conv_messages = []
    max_message_length = 50000
    for msg in messages:
        if msg.content and len(msg.content) > max_message_length:
            msg.content = msg.content[:max_message_length]
        role = msg.role
        logger.debug(f"Converting input message of type {role}")
        if role.lower() == 'user' or role.lower() == 'human':
            new_message = HumanMessage(content=msg.content)
        if role.lower() == 'system':
            new_message = SystemMessage(content=msg.content)
        if role.lower() == 'assistant':
            content = ''
            additional_kwargs = {}
            if msg.content:
                content = msg.content
            if msg.tool_calls:
                # Convert list of AIToolCall messages to langchain ToolCall message
                langchain_tool_calls = []
                for index, tool_call in enumerate(msg.tool_calls):
                    name = tool_call.function.name
                    args = tool_call.function.arguments
                    id = tool_call.id
                    langchain_tool_calls.append(ToolCall(name=name, args=args, id=id, type='tool'))

                new_message = AIMessage(content=content, tool_calls=langchain_tool_calls, additional_kwargs=additional_kwargs)
            else:
                new_message = AIMessage(content=content, additional_kwargs=additional_kwargs)
        if role.lower() == 'tool':
            tool_call_id = msg.tool_call_id
            content = msg.content
            name = None
            new_message = ToolMessage(content=content, name=name, tool_call_id=tool_call_id)
        conv_messages.append(new_message)
    return {
        "messages": conv_messages
    }

def convert_response_to_messages(response: dict) -> List[Message]:
    messages = []
    for msg in response['messages']:
        role = 'not found'
        if msg.type:
            role = msg.type
        logger.info(f"Processing role {role}")
        tool_calls = None
        if 'tool_calls' in msg:
            tool_calls = msg['tool_calls']
        if msg.additional_kwargs:
            additional_kwargs = msg.additional_kwargs
            if 'tool_calls' in additional_kwargs:
                tool_calls = []
                for tool_call_data in additional_kwargs['tool_calls']:
                    function_arguments = tool_call_data['function']['arguments']
                    if isinstance(function_arguments, str):
                        function_arguments = json.loads(function_arguments)
                    tool_call = AIToolCall(
                        id=tool_call_data['id'],
                        function=Function(
                            arguments=function_arguments,
                            name=tool_call_data['function']['name']
                        ),
                        type=tool_call_data['type']
                    )
                    tool_calls.append(tool_call)
        content = ""
        if msg.content:
            content = msg.content 
        id = None
        if msg.id:
            id = msg.id 
        name = None
        if 'name' in msg:
            name = msg['name']
        if msg.name:
            name = msg.name
        tool_call_id = None
        if 'tool_call_id' in msg:
            tool_call_id = msg['tool_call_id'] 
        if role == 'tool' and msg.tool_call_id:
            tool_call_id = msg.tool_call_id
        if role == 'human':
            message = Message(
            role='user',
            content=content
            )
        elif role == 'ai':
            message = Message(
            role='assistant',
            content=content
            )
        else:
            message = Message(
                role=role,
                content=content
            )
        messages.append(message)
    return messages

def get_llm_sync(messages: List[Message], model: str, thread_id: str, tools):
    logger.info(f"LLM Synchronous call using model {model} and tools {tools}")
    model_instance = None
    if 'gpt' in model:
        if not OPENAI_API_KEY:
            return "API key not set\n"
        model_instance = init_openai(model, {})
    else:
        client_model_instance = None
        if WATSONX_SPACE_ID:
           client_model_instance = APIClient(credentials=Credentials(url=WATSONX_URL, token=get_access_token(WATSONX_API_KEY)),
                       space_id=WATSONX_SPACE_ID)
        elif WATSONX_PROJECT_ID:
            client_model_instance = APIClient(credentials=Credentials(url=WATSONX_URL, token=get_access_token(WATSONX_API_KEY)),
                       project_id=WATSONX_PROJECT_ID)
        else:
            logger.error("You must either set WATSONX_SPACE_ID or WATSONX_PROJECT_ID")
        model_instance = ChatWatsonx(model_id=model, watsonx_client=client_model_instance)
    logger.info(f"Starting with input messages: {messages}")
    inputs = convert_messages_to_langgraph_format(messages)
    validate_chat_history(inputs["messages"])
    logger.info(f"Calling langgraph with input: {inputs}")
    if tools:
       graph = create_react_agent(model_instance, tools=tools)
       response = graph.invoke(inputs)
    else:
        graph = model_instance
        response = graph.invoke(inputs['messages'])
    logger.info(f"Response: {response}")
    if hasattr(response, 'content'):
        results = response.content
        message = Message(
                role='ai',
                content=results
            )
        response_messages = [message.dict()]
    else:
        results = response["messages"][-1].content
    return results, messages

def format_resp(struct):
    return "data: " + json.dumps(struct) + "\n\n"

def validate_chat_history(messages: List[BaseMessage]):
    tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if isinstance(tool_call, dict):
                    tool_call_ids.add(tool_call.get('id'))
                else:
                    tool_call_ids.add(tool_call.id)

    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in tool_call_ids:
                tool_call_ids.remove(msg.tool_call_id)

    for tool_call_id in tool_call_ids:
        logger.info(f"Fixing input that had no tool response for tool_call_id {tool_call_id}")
        placeholder_message = ToolMessage(
            content="Tool call failed or no response received.",
            tool_call_id=tool_call_id,
            name="unknown"
        )
        messages.append(placeholder_message)

async def get_llm_stream(messages: List[Message], model: str, thread_id: str, tools):
    if tools:
        use_tools = True
    else:
        use_tools = False
    send_tool_events = True
    logger.info(f"LLM Stream with tools {tools}")
    model_init_overrides = {'temperature': 0, 'streaming': True}
    if not thread_id:
        logger.warn("Warning no thread_id specified in input")
        thread_id = ""
    if 'gpt' in model:
        if not OPENAI_API_KEY:
            yield "API key not set\n"
        model_instance = init_openai(model, model_init_overrides)
    else:
        client_model_instance = None
        if WATSONX_SPACE_ID:
           client_model_instance = APIClient(credentials=Credentials(url=WATSONX_URL, token=get_access_token(WATSONX_API_KEY)),
                       space_id=WATSONX_SPACE_ID)
        elif WATSONX_PROJECT_ID:
            client_model_instance = APIClient(credentials=Credentials(url=WATSONX_URL, token=get_access_token(WATSONX_API_KEY)),
                       project_id=WATSONX_PROJECT_ID)
        else:
            logger.error("You must either set WATSONX_SPACE_ID or WATSONX_PROJECT_ID")
        model_instance = ChatWatsonx(model_id=model, watsonx_client=client_model_instance)
    if use_tools:
        graph = create_react_agent(model_instance, tools=tools)
    else:
        graph = create_react_agent(model_instance, tools=[])
    inputs = ""
    accumulated_contents = ""
    try:
        inputs = convert_messages_to_langgraph_format(messages)
        validate_chat_history(inputs["messages"])
        async for event in graph.astream_events(inputs, version="v2"):
            kind = event["event"]
            logger.debug(f"event = {event}")
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    if isinstance(content, str):
                        current_timestamp = int(time.time())
                        struct = {
                            "id": str(uuid.uuid4()),
                            "object": "thread.message.delta",
                            "created": current_timestamp,
                            "thread_id": thread_id,
                            "model": model,
                            "choices": [
                                {
                                    "delta": {
                                        "content": content,
                                        "role": "assistant",
                                    }
                                }
                            ],
                        }
                        event_content = format_resp(struct)
                        logger.debug("Sending event content: " + event_content)
                        accumulated_contents += content
                        yield event_content
                    elif isinstance(content, list):
                        for item in content:
                            if 'type' in item:
                                if item['type'] == 'text':
                                    yield item['text']
                                elif item['type'] == 'tool_use':
                                    logger.debug("tool_use")
                                    logger.debug(f"{str(item)}")
                                else:
                                    logger.debug("Received item of type " + item['type'])
            elif kind == "on_tool_start":
                printmsg =  f"Starting tool: {event['name']} with inputs: {event['data'].get('input')} run_id: {event['run_id']}"
                logger.debug(printmsg)
                current_timestamp = int(time.time())
                step_details = {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": event['run_id'],
                            "name": event['name'],
                            "args": event['data'].get('input')
                        }
                    ]
                }
                struct = {
                            "id": str(uuid.uuid4()),
                            "object": "thread.run.step.delta",
                            "thread_id": thread_id,
                            "model": model,
                            "created": current_timestamp,
                            "choices": [
                                {
                                    "delta": {
                                        "role": "assistant",
                                        "step_details": step_details
                                    }
                                }
                            ],
                         }
                event_content = format_resp(struct)
                logger.info("Sending tool call event content: " + event_content)
                if send_tool_events:
                    yield event_content
            elif kind == "on_tool_end": 
                tool_name = event.get('name', '')
                logger.info(f"Event on_tool_end for tool: {tool_name}")
                output = event.get('data', {}).get('output', {})
                content = ''
                if output and output.content:
                    content = output.content
                run_id = event['run_id']      
                logger.info(f"Tool output for run {run_id} was: {content}")
                tool_call_id = ''
                if output and output.tool_call_id:
                    tool_call_id = output.tool_call_id
                tool_call_id = run_id #Better matches tool response with tool request
                current_timestamp = int(time.time())
                step_details = {
                    "type": "tool_response",
                    "name": event['name'],
                    "tool_call_id": tool_call_id,
                    "content": content
                }
                struct = {
                            "id": str(uuid.uuid4()),
                            "object": "thread.run.step.delta",
                            "thread_id": thread_id,
                            "model": model,
                            "created": current_timestamp,
                            "choices": [
                                {
                                    "delta": {
                                        "role": "assistant",
                                        "step_details": step_details
                                    }
                                }
                            ],
                         }
                event_content = format_resp(struct)
                logger.info("Sending tool response event content: " + event_content)
                if send_tool_events:
                    yield event_content
            elif kind == "on_chat_model_start": 
                logger.debug(f"Received event type: on_chat_model_start")
            elif kind == "on_chat_model_end": 
                logger.debug(f"Received event type: on_chat_model_end")
            else:
                logger.debug("Received event type: " + kind)
            yield ""

        if accumulated_contents:
            logger.info("Final streamed content:\n" + accumulated_contents)

    except Exception as e:
        logger.error(f"Exception {str(e)}")
        traceback.print_exc()
        logger.error(f"Exception was with inputs {str(inputs)}")
        yield f"Error: {str(e)}\n"
