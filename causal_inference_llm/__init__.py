# SPDX-FileCopyrightText: Copyright © 2024 Patricio Jaime Porras <contact@patriciojaime.dev>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from lunarcore.core.typings.components import ComponentGroup
from lunarcore.core.data_models import ComponentInput, ComponentModel
from lunarcore.core.typings.datatypes import DataType
from lunarcore.core.component import BaseComponent
from langchain_openai import ChatOpenAI

from typing import List, Optional, Any
import pandas as pd

from .causal_inference_llm import CausalInferenceAgentLLM

def get_input_value_by_name(input_name: str, inputs: List[ComponentInput], default=None) -> Any:
    for inp in inputs:
        if inp.key == input_name:
            return inp.value
    return default

def get_object_from_input(input_name: str, dict_input: dict, default=None) -> Any:
    if input_name in dict_input:
        return dict_input[input_name]
    return default

class CausalInferenceLLM(
    BaseComponent,
    component_name="Causal Inference with a LLM",
    component_description="""
    Run Causal Inference with the help of a LLM to run different methods (DoWhy and CausalPy)
    
    Inputs:
        Data Path (str): Path to the file
        Data Separator (str): Separator of the file
        Background Graph (str): Node-Link formatted Graph
        Context (str): Context for the causal inference methods
        Log File Name (str): Name of the log file to store the step-by-step reasoning of the LLM and results
        
    Outputs:
        Results (JSON):
            - Agent output (str): Output of the LLM
            - Dowhy results (str): Results of the causal inference methods
            - CausalPy results (str): Results of the causal inference methods
    """,
    input_types={
        "Data Path": DataType.TEXT,
        "Data Separator": DataType.TEXT,
        "Background Graph": DataType.JSON,
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
        try:
            data_path = get_input_value_by_name("Data Path", inputs)
            sep = get_input_value_by_name("Data Separator", inputs)
            df = pd.read_csv(data_path, sep=sep)
        except Exception as e:
            raise Exception(f"Error getting dataset: {e}")
        
        background_knowledge = get_input_value_by_name("Background Graph", inputs)
        context = get_input_value_by_name("Context", inputs)
        file_name = get_input_value_by_name("Log File Name", inputs)
        
        
        cdAgent = CausalInferenceAgentLLM(
            client=self._client,
            data=df,
            bg_knowledge=background_knowledge,
            )
        
        result = cdAgent.determine_causal_inference(context=context)
        out = result['agent_output']
        dowhy = result['dowhy']
        causalpy = result['causalpy']
        full_log = result['full_log']
        
        try:
            self._file_connector.delete_file(f"{file_name}.txt")
        except:
            pass
        self._file_connector.create_file(file_name=f"{file_name}.txt", content=full_log)
         
        return {
            "agent": str(out),
            "dowhy": str(dowhy),
            "causalpy": str(causalpy)
        }