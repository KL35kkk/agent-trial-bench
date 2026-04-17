"""
Agent Runner for Agent Trial Bench
Handles Agent API calls and task execution
"""

import asyncio
import httpx
from typing import Dict, Any, Optional
import logging

from .base import BaseRunner

logger = logging.getLogger(__name__)


class AgentRunner(BaseRunner):
    """Runner that handles Agent API calls.
    
    This runner is responsible for:
    - Calling Agent APIs
    - Managing Agent connections
    - Handling timeouts and errors
    """
    
    def __init__(self, config: Dict[str, Any] = None, debug: bool = False):
        super().__init__(config, debug)
        self.timeout = self.config.get('timeout', 30)
    
    async def call_agent_api(self, 
                            url: str, 
                            question: str, 
                            context: Optional[Dict[str, Any]] = None,
                            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Call Agent API.
        
        Args:
            url: Agent API URL
            question: Question to ask
            context: Context information
            timeout: Request timeout (uses self.timeout if not provided)
            
        Returns:
            Agent response dictionary
        """
        timeout = timeout or self.timeout
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                request_data = {
                    "question": question,
                    "context": context or {}
                }
                
                response = await client.post(
                    f"{url}/process",
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Agent API returned status {response.status_code}: {response.text}")
                
                return response.json()
                
        except httpx.TimeoutException:
            raise Exception(f"Agent API call timed out after {timeout}s")
        except httpx.RequestError as e:
            raise Exception(f"Agent API request failed: {e}")
        except Exception as e:
            logger.error(f"Failed to call agent API: {e}")
            raise
    
    async def close_agent(self, close_endpoint: Optional[str]):
        """Close Agent connection if close_endpoint is provided"""
        if not close_endpoint:
            return
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    close_endpoint,
                    json={"reason": "Task completed"}
                )
                logger.debug(f"Agent closed at {close_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to close agent: {e}")
    
    async def launch(self, tasks: list) -> list:
        """
        Launch tasks (not used for AgentRunner, use call_agent_api directly)
        
        This method is kept for interface compatibility but AgentRunner
        is typically used directly via call_agent_api.
        """
        logger.warning("AgentRunner.launch() is not typically used. Use call_agent_api() directly.")
        return []

