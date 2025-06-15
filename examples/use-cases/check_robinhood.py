# Goal: Checks for available visa appointment slots on the Greece MFA website.

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')

max_steps = 5
max_actions_per_step = 1
ROBINHOOD_USERNAME = os.getenv("ROBINHOOD_USERNAME")
ROBINHOOD_PASSWORD = os.getenv("ROBINHOOD_PASSWORD")

controller = Controller()


class WebpageInfo(BaseModel):
	"""Model for webpage link."""
	link: str = 'https://robinhood.com'


@controller.action('Go to the webpage', param_model=WebpageInfo)
def go_to_webpage(webpage_info: WebpageInfo):
	"""Returns the webpage link."""
	return webpage_info.link


async def main():
	"""Main function to execute the agent task."""
	task = f"""
		You are logged in to Robinhood. Goto Account menu and then Investing sub-menu and Return me the details of the cryptocurrencies and stock from the right section of the screen.
		Important: 
		1. Return the result in a json format.
		2. The json should have the following fields:
			- symbol
			- name
			- quantity
			- average price
			- total value
			- change percent
		3. The json should be in the following format:
		{{
			"cryptocurrencies": [
				{{
					"symbol": "BTC",
					"name": "Bitcoin",
					"quantity": 1,
					"average price": 10000,
					"total value": 10000,
					"change percent": 10
				}}	
			],
			"stocks": [
				{{
					"symbol": "AAPL",
					"name": "Apple",
					"quantity": 1,
					"average price": 10000,
					"total value": 10000,
					"change percent": 10
				}}
			]
		}}
		4. Do not return any other text or comments.
"""
	task = task.strip()

	model = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(os.getenv('OPENAI_API_KEY', '')))
	agent = Agent(task, model, controller=controller, use_vision=True, max_actions_per_step=max_actions_per_step)

	history = await agent.run(max_steps=max_steps)
	result = history.final_result()
	if result:
		print(" extracted result")
		print(result)

	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())
