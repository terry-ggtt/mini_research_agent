from datetime import datetime
from pathlib import Path
from tavily import TavilyClient
import os
from typing_extensions import Literal , List , Sequence
def get_today_str()->str:
    today = datetime.now()
    return f"{today:%a %b} {today.day},{today:%Y}"

def get_current_dir()->Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def create_research_graph(model=None):

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def tavily_search_multiple(
            search_querier:List[str],
            max_results:int = 3 ,
            topic:Literal["general","news","finance"] = "general" ,
            include_raw_content:bool = True
    )->List[dict]:
        search_docs = []
        for query in search_querier:
            result = tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic
            )
            search_docs.append(result)
        return search_docs

    
    