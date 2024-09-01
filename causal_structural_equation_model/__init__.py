# SPDX-FileCopyrightText: Copyright © 2024 Patricio Jaime Porras <contact@patriciojaime.dev>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from lunarcore.core.typings.components import ComponentGroup
from lunarcore.core.data_models import ComponentInput, ComponentModel
from lunarcore.core.typings.datatypes import DataType
from lunarcore.core.component import BaseComponent

from langchain.chat_models import ChatOpenAI

from .sem_llm import SEMAgentLLM, SEMEnvironment

from typing import List, Optional, Any
import pandas as pd

def get_input_value_by_name(input_name: str, inputs: List[ComponentInput], default=None) -> Any:
    for inp in inputs:
        if inp.key == input_name:
            return inp.value
    return default


class StructuralEquationModel(
    BaseComponent,
    component_name="Structural Equation Model Refinement with LLM",
    component_description="Run SemoPy with an initial SEM so it can be refined and interpreted by an LLM.",
    input_types={
        "Data": DataType.TEXT,
        "Data Separator": DataType.TEXT,
        "SEM": DataType.TEXT,
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
        
    def run(self, inputs: List[ComponentInput],**kwargs: Any):
        data_path = get_input_value_by_name("Data", inputs)
        sem_init_model = get_input_value_by_name("SEM", inputs)
        context = get_input_value_by_name("Context", inputs)
        sep = get_input_value_by_name("Data Separator", inputs)
        file_name = get_input_value_by_name("Log File Name", inputs)
        
        try:
            df = pd.read_csv(data_path, header=0, sep=sep)
        except Exception as e:
            raise Exception(f"Error getting dataset: {e}")
        
        sem_env = SEMEnvironment(sem_init_model, df)
        sem_llm = SEMAgentLLM(self._client, sem_env)
        result = sem_llm.improve_sem_model(context)
        log = result['full_log']
        try:
            self._file_connector.delete_file(f"{file_name}.txt")
        except:
            pass
        self._file_connector.create_file(file_name=f"{file_name}.txt", content=log)
        
        return {
            'output': result['agent_output'],
            'interpretation': result['interpretation'],
            'model': result['final_model'],
        }