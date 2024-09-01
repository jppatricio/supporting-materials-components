# SPDX-FileCopyrightText: Copyright © 2024 Patricio Jaime Porras <contact@patriciojaime.dev>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from lunarcore.core.typings.components import ComponentGroup
from lunarcore.core.data_models import ComponentInput, ComponentModel
from lunarcore.core.typings.datatypes import DataType
from lunarcore.core.component import BaseComponent

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

from .build_causal_digraph import CausalGraphBuilder
from .causal_relation_llm import CausalDiscoveryAgentLLM

from typing import List, Optional, Any
import pandas as pd
from itertools import combinations

def get_input_value_by_name(input_name: str, inputs: List[ComponentInput], default=None) -> Any:
    for inp in inputs:
        if inp.key == input_name:
            return inp.value
    return default


class CausalGraphDiscovery(
    BaseComponent,
    component_name="Causal Graph Discovery with LLM",
    component_description="""
    Run Causal Graph Discovery thrugh a LLM with access to wikipedia (Agent)
    
    It will run a LLM with the following steps:
    1. Query Wikipedia to find a causal relation between each pair of variables in the data
    2. Run a LLM to find if there is a direct causation between the variables
    3. The LLM will output a causal graph with the causal relations found
        Plus a log of the final causal result (if A->B, B->A, or if there was no causal relation)
        > Full log on the LLM step-by-step process will be saved as a txt file with the name provided
        
    Inputs:
        Data (str): Path to the file
        Data Separator (str): Separator of the file
        Context (str): Context for causal discovery (hypothesis generation)
        Log File Name (str): Name of the log file to store the step-by-step reasoning of the LLM

    Outputs:
        Results (JSON):
            - The found causal graph (node-link format)
            - Agent output
    """,
    input_types={
        "Data": DataType.TEXT,
        "Data separator": DataType.TEXT,
        "Context": DataType.TEXT,
        "Log File Name": DataType.TEXT,
        },
    output_type=DataType.ANY,
    component_group=ComponentGroup.CAUSAL_INFERENCE,
    openai_api_key=None,
    model_name="gpt-3.5-turbo",
):
    def __init__(self, model: Optional[ComponentModel] = None, **kwargs: Any):
        super().__init__(model=model, configuration=kwargs)
        self._client = ChatOpenAI(**self.configuration)
        
    def run(self, inputs: List[ComponentInput], **kwargs: Any):
        # Get the inputs
        data_path = get_input_value_by_name("Data", inputs)
        sep = get_input_value_by_name("Data separator", inputs)
        context = get_input_value_by_name("Context", inputs)
        file_name = get_input_value_by_name("Log File Name", inputs)
        
        # Read the CSV file
        try:
            df = pd.read_csv(data_path, sep=sep)
        except Exception as e:
            raise Exception(f"Error getting dataset: {e}")
        
        # Run the causal discovery
        causal_discovery = CausalDiscoveryAgentLLM(self._client, WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
        result_log = []
        full_log = []
        graph_builder = CausalGraphBuilder()
        # Look for a->b or b->a for (a,b) in comb(vars, 2)
        for var_1, var_2 in list(combinations(df.columns, 2)):
            result = causal_discovery.determine_causal_relationship(var_1, var_2, context)
            pred = result['prediction']
            # save the logs
            result_log.append(f"{result['agent_output']} -> {pred}")
            full_log.append(result['full_log'])
            # update the graph
            graph_builder.update_graph(var_1, var_2, pred)
            
        # save the log file
        try:
            self._file_connector.delete_file(f"{file_name}.txt")
        except:
            pass
        self._file_connector.create_file(file_name=f"{file_name}.txt", content="\n".join(full_log))
        
        return {
            'graph': graph_builder.get_graph(),
            'log': result_log,
            }