# AI Workflow Automator

This project is a LangGraph-based workflow automator that uses natural language commands to summarize calendar events, send messages to Slack, and log updates to Notion.

Example command:
> Summarize today’s schedule and send it to Slack and Notion.

## Prerequisites
If you have access to the FurisoaAI LLM API endpoint or a dedicated RNGD server, the only step you need to take before beginning the SDK cookbook is installing the necessary prerequisites for the FuriosaAI SDK stack. Please refer to the provided page to choose and set up your preferred version of the FuriosaAI SDK. Currently, this SDK cookbook is based on FuriosaAI SDK version 2025.3.0.
- [Install Prerequisites for FuriosaAI SDK](https://developer.furiosa.ai/latest/en/get_started/prerequisites.html)

## Set Environments 

### Requirements
- FuriosaSDK 2025.3.0.
- RNGD server or Endpoint API

### Installation
```
   git clone https://github.com/furiosa-ai/sdk-cookbook.git
   cd workflow-automation
   python -m venv venv
   source /venv/bin/activate
   pip install -r requirements.txt
```

## Configuration
- LLM:
   - [furiosa-ai/Llama-3.1-8B-Instruct](https://huggingface.co/furiosa-ai/Llama-3.1-8B-Instruct) 

## Usage

Execute the `.py` file using the command below to test the recipe for building an AI Workflow Automator that connects calendar data with Slack and Notion through natural language commands.  
This project applies the concept of workflow orchestration using LangGraph to automatically summarize, post, and log daily schedules.

You can enter any natural language instruction (e.g., “Summarize today’s schedule and send it to Slack and Notion”),  
and the workflow will automatically plan, fetch, summarize, and execute the required actions.

```
python main.py
```

## File Structure

```
workflow_automation/
│
├── main.py                  # Main execution file
├── requirements.txt          # Dependencies list
├── .env                      # Environment variables
└── README.md                 # Project documentation
