from pydantic import BaseModel, Field, ConfigDict, TypeAdapter
from typing import Optional
from nltk.corpus import wordnet
import asyncio, json, re
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


### BEGIN Pydantic classes ------------ 
class WordSense(BaseModel):
    """Word definition."""
    sense: Optional[str] = Field(description="Sense definition of the specified word")
    probability: Optional[float] = Field(
        default=None, description="How likely in the scale of 0 to 1, does the definition accurately represent the sense?"
    )
    model_config = ConfigDict(extra='ignore')

class WordNetBasedSenseOutput(BaseModel):
    word: Optional[str] = Field(description="input term")
    senses: Optional[list[str]] = Field(description="input list of senses")
    human_value_senses: Optional[list[str]] = Field(description="In the input list of senses, identify all the senses that represents a core value or belief")
    probability: Optional[float] = Field(
        default=None, description="How likely in the scale of 0 to 1, do the identified senses connote human values?"
    )
    model_config = ConfigDict(extra='ignore')

# Format word sense for 
class MiscSenseOutput(BaseModel):
    word: Optional[str] = Field(description="input term")
    human_value_senses: Optional[list[str]] = Field(description="Identify all the senses that represents a human value, core values or belief")
    probability: Optional[float] = Field(
        default=None, description="How likely in the scale of 0 to 1, do the identified senses connote human values?"
    )
    model_config = ConfigDict(extra='ignore')

class WordNetBasedNonValueSenseOutput(BaseModel):
    word: Optional[str] = Field(description="input term")
    senses: Optional[list[str]] = Field(description="input list of senses")
    non_value_senses: Optional[list[str]] = Field(description="In the input list of senses, identify all the senses that do not represent a human value")
    model_config = ConfigDict(extra='ignore')

# Format word sense for 
class MiscNonValueSenseOutput(BaseModel):
    word: Optional[str] = Field(description="input term")
    non_value_senses: Optional[list[str]] = Field(description="Identify at most five senses that do not represent a human value, core values or belief")
    model_config = ConfigDict(extra='ignore')

class NounAdjSense(BaseModel):
    word: str = Field(description="noun term")
    sense_definition: str = Field(description="sense definition that the word connotes in the sentence")
    part_of_speech: str = Field(description="part of speech of the word, either noun or adjective")
    model_config = ConfigDict(extra='ignore')

class NounAdjSensesOutput(BaseModel):
    sentence: str = Field(description="input sentence")
    terms: Optional[list[NounAdjSense]] = Field(description="list of nouns in the sentence and the sense definitions of each noun")
    model_config = ConfigDict(extra='ignore')

class OppositeTerm(BaseModel):
    term: str = Field(description="antonym or opposite of the input term")
    model_config = ConfigDict(extra='ignore')

class HumanValueJustificationOutput(BaseModel):
    sentence: str = Field(description="input text that describes a sense")
    justification: str = Field(description="reason why the given text connotes a human value")
    model_config = ConfigDict(extra='ignore')

### END Pydantic classes ------------ 

def setup_model(model_name):
    ollama_model = OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        settings=ModelSettings(temperature=0.0),
    )
    return ollama_model

## Define method for LLM to call by iteself -- Does not work reliably, as the LLM sometimes does not call it.
def tool_get_wordnet_senses(ctx: RunContext[int]) -> list:
    senses = []
    input_word = ctx.deps
    print('Called get_word_senses for ', input_word)
    synsets = wordnet.synsets(input_word)
    if len(synsets) > 0:
        for syn in synsets:
            senses.append(syn.definition())
    return senses

## Get sense definitions (glosses) from wordnet
def get_wordnet_glosses(word):
    senses = []
    synsets = wordnet.synsets(word)
    if len(synsets) > 0:
        for syn in synsets:
            senses.append(syn.definition())
    return senses

def sanitize_string(res_str):
    res_str = re.sub(r'\n+', ' ', res_str)
    res_str = re.sub('\\\\', '', res_str)
    res_str = re.sub('\'', '', res_str)
    res_str = re.sub('`+', '', res_str)
    res_str = re.sub('\\*+', '', res_str)
    res_str = re.sub(r'\s{2,}', ' ', res_str)
    res_str = re.sub(r'Page.*\d+.*ethereum.org', '', res_str)
    res_str = re.sub(r'http.*\s', '', res_str)
    return res_str

def read_lines_from_file(file_path, lines_to_read=20, start_line_number=1):
    read_lines = []
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, 1):  # start enumeration from 1
                line = sanitize_string(line)
                #print(f"read: {line}")
                if i >= start_line_number and i < start_line_number + lines_to_read:
                    read_lines.append(line)
                elif i >= start_line_number + lines_to_read:
                    break  # Stop after reading the desired number of lines
        return read_lines
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def append_to_file(data, file_path):
    try:
        with open(file_path, 'a') as f:
            json.dump(data, f, indent=4)
            f.write("\n")
    except FileNotFoundError:
        print(f"Error: '{file_path}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' contains invalid JSON. Overwriting.")

def setup_agent_predict_wordnet_based_hv_senses(model_name):
    model = setup_model(model_name)
    rag_agent = Agent(
            model=model, 
            system_prompt=(
                """
                You are a helpful assistant. Human values are core values or beliefs. In the input list of senses, identify all the senses that connote human values. 
                Also estimate how likely do the identified senses connote human values. Do not summarize or rephrase any senses. Remove any enumerations at the beginning of sense definitions.
                """
            ),
            output_type=WordNetBasedSenseOutput,
            output_retries=3)
    return rag_agent

def setup_agent_predict_misc_hv_senses(model_name):
    model = setup_model(model_name)
    rag_agent = Agent(
            model=model, 
            system_prompt=(
                """
                You are a helpful assistant. Human values are core values or beliefs. Your task is to define all 
                possible senses that connote human values for the input terms. Remove any enumerations at the beginning of sense definitions. 
                Also estimate how likely do the identified senses connote human values.
                """
            ),
            output_type=MiscSenseOutput,
            output_retries=3)
    return rag_agent

def setup_agent(model_name, prompt, output_type):
    model = setup_model(model_name)
    rag_agent = Agent(
            model=model, 
            system_prompt=(
                prompt
            ),
            output_type=output_type,
            output_retries=3)
    return rag_agent

#############################
def prompt_predict_wordnet_based_hv_senses(rag_agent, input_term, input_list_senses):
    user_prompt_str = f"""
                Human values connote what is important to us in life. Human values are beliefs that when activated, become infused with feeling. Human Values refer to desirable goals that motivate action. Human values transcend specific actions and situations. Human values serve as standards or criteria to guide the selection or evaluation of actions, policies, people, and events.
                You are a helpful assistant. Your task is to identify all the senses that connote human values in the input list of senses. Follow the example.

                term: power
                senses: ['possession of controlling influence', '(physics) the rate of doing work; measured in watts (= joules/second)', 'possession of the qualities (especially mental qualities) required to do something or get something done', '(of a government or government official) holding an office means being in power', 'one possessing or exercising power or influence or authority', 'a mathematical notation indicating the number of times a quantity is multiplied by itself', 'physical strength', 'a state powerful enough to influence events throughout the world', 'a very wealthy or powerful businessman', 'supply the force or power for the functioning of']
                human_value_senses: ['possession of controlling influence', 'possession of the qualities (especially mental qualities) required to do something or get something done', 'one possessing or exercising power or influence or authority', 'physical strength']
                probability: '0.5'
                term: {input_term}
                senses: {input_list_senses}
                human_value_senses: 
                probability: 
                """

    result = rag_agent.run_sync(user_prompt_str)
    return result

def prompt_predict_misc_hv_senses(rag_agent, input_term):
    user_prompt_str = f"""
                Define all possible senses that connote human values for the input terms. Follow the example.

                term: 'a comfortable life'
                human_value_senses: ['a lifestyle characterized by ease, security, and freedom from want or hardship, with sufficient resources and pleasant conditions']
                probability: '0.5'
                term: {input_term}
                human_value_senses: 
                probability: 
                """
    result = rag_agent.run_sync(user_prompt_str)
    return result

def prompt_predict_wordnet_based_nonvalue_senses(model_name, input_term, input_list_senses):
    prompt_str = f"""
        Human values connote what is important to us in life. Human values are beliefs that when activated, become infused with feeling. Human Values refer to desirable goals that motivate action. Human values transcend specific actions and situations. Human values serve as standards or criteria to guide the selection or evaluation of actions, policies, people, and events. 
        You are a helpful assistant. Your task is to identify all the senses, the input list of senses, which do not connote human values. Follow the example.

        term: power
        senses: ['possession of controlling influence', '(physics) the rate of doing work; measured in watts (= joules/second)', 'possession of the qualities (especially mental qualities) required to do something or get something done', '(of a government or government official) holding an office means being in power', 'one possessing or exercising power or influence or authority', 'a mathematical notation indicating the number of times a quantity is multiplied by itself', 'physical strength', 'a state powerful enough to influence events throughout the world', 'a very wealthy or powerful businessman', 'supply the force or power for the functioning of']
        non_value_senses: ['(physics) the rate of doing work; measured in watts (= joules/second)', '(of a government or government official) holding an office means being in power', 'a mathematical notation indicating the number of times a quantity is multiplied by itself', 'a state powerful enough to influence events throughout the world', 'a very wealthy or powerful businessman', 'supply the force or power for the functioning of']
        term: {input_term}
        senses: {input_list_senses}
        non_value_senses: 
    """
    output_type = WordNetBasedNonValueSenseOutput
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result

def prompt_predict_misc_nonvalue_senses(model_name, input_term):
    prompt_str = f"""
        Human values connote what is important to us in life. Human values are beliefs that when activated, become infused with feeling. Human Values refer to desirable goals that motivate action. Human values transcend specific actions and situations. Human values serve as standards or criteria to guide the selection or evaluation of actions, policies, people, and events. 
        You are a helpful assistant. Your task is to define at most five senses that do not connote human values for the input terms. Follow the example.

        term: a sense of futility
        non_value_senses: ['the feeling of pointlessness or uselessness, the belief that an action or effort will have no result or effect', 'The feeling can emerge from being confronted with large-scale problems or tragedies, such as war or personal hardship']
        term: {input_term}
        non_value_senses: 
    """
    output_type = MiscNonValueSenseOutput
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result
    

def prompt_infer_noun_adjective_sense(model_name, sentence):
    prompt_str = f"""
        You are a helpful assistant. Your task is to identify all nouns and adjectives in the input sentence and 
        define the sense that each identified word best connotes in the given sentence. Do not include proper nouns. Remove any enumerations at the beginning of sense definitions. 
        Follow the example:

        'sentence': The purpose of this is to make block creation computationally 'hard', thereby preventing sybil attackers from remaking the entire blockchain in their favor.
        'terms': [
             {{ 'word': 'purpose', 'sense_definition': 'an anticipated outcome that is intended or that guides your planned actions', 'part_of_speech': 'noun' }},
             {{ 'word': 'block', 'sense_definition': 'a digital container that holds a set of validated transactions, a timestamp, and a cryptographic hash that links it to the previous block in the chain', 'part_of_speech': 'noun' }},
             {{ 'word': 'hard', 'sense_definition': 'requiring great physical or mental effort to accomplish or comprehend or endure', 'part_of_speech': 'adjective' }},
             {{ 'word': 'creation', 'sense_definition': 'introducing something new', 'part_of_speech': 'noun' }},
             {{ 'word': 'sybil', 'sense_definition': 'a single malicious entity creates a large number of fake identities (nodes) to gain disproportionate influence over the network, disrupt its consensus, and potentially launch other attacks', 'part_of_speech': 'noun' }},
             {{ 'word': 'attackers', 'sense_definition': 'a malicious entities aiming to compromise the network\'s integrity and security', 'part_of_speech': 'noun' }},
             {{ 'word': 'entire', 'sense_definition': 'constituting the full quantity or extent; complete', 'part_of_speech': 'adjective' }},
             {{ 'word': 'blockchain', 'sense_definition': 'a system in which a record of transactions, especially those made in a cryptocurrency, is maintained across computers that are linked in a peer-to-peer network', 'part_of_speech': 'noun' }},
             {{ 'word': 'favor', 'sense_definition': 'an advantage to the benefit of someone or something', 'part_of_speech': 'noun' }}
        ]
        'sentence': {sentence}
        'terms':
    """
    output_type = NounAdjSensesOutput
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result

def prompt_infer_compund_sense(model_name, sentence):
    prompt_str = f"""
        You are a helpful assistant. Your task is to identify all compound words separed by hyphen (-) in the input sentence and 
        define the sense that each compound word best connotes in the given sentence. Examples of compound words are mind-blowing, and check-in.
        Remove any enumerations at the beginning of sense definitions. If no compound word exists, return an empty list.
        Follow the example:

        'sentence': And so, we can use ZK-SNARKs to greatly simplify privacy-preserving and improve privacy in proof-of-liabilities protocols.
        'terms': [
             {{ 'word': 'ZK-SNARKs', 'sense_definition': 'a cryptographic protocol', 'part_of_speech': 'noun' }},
             {{ 'word': 'privacy-preserving', 'sense_definition': 'methods and technologies that protect sensitive user data while still allowing valuable analysis, training models, or computations', 'part_of_speech': 'noun' }},
             {{ 'word': 'proof-of-liabilities', 'sense_definition': 'a cryptographic method letting companies (like crypto exchanges) prove they hold enough assets to cover customer funds (liabilities) without revealing private user data', 'part_of_speech': 'noun' }},
        ]
        'sentence': {sentence}
        'terms':
    """
    output_type = NounAdjSensesOutput
    #print(prompt_str)
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result


def prompt_infer_human_value(model_name, input_sense):
    prompt_str = f"""
    You are a helpful assistant. Your task is to justify why the given text connotes a human value. Select the most relevant justification from the following:
    1. It describes a belief that, when activated, becomes emotionally charged, influencing a person's feelings and reactions, such as distress when the value is threatened or joy when it is realized.
    2. It implies a desirable goal that motivate action, such as pursuing social order, justice, and helpfulness, because they represent what an individual finds important.
    3. It describes a general principle that is relevant across diverse actions and situations, distinguishing it from more context-specific norms and attitudes.
    4. It represents an unconscious standard or criterion that guides choices and judgments about actions, policies, and people, only rising to conscious awareness when a different conflicting value make a decision difficult.
    5. It represents one of many priorities that characterize a person.
    6. It describes a tradeoff among multiple, often competing, values, influencing choices based on which values are activated and most important to the individual.
    Follow the examples below:

    'sense': 'an anticipated outcome that is intended or that guides your planned actions'
    'justification': 'It implies a desirable goal that motivate action, such as pursuing social order, justice, and helpfulness, because they represent what an individual finds important.'
    'sense': 'a single malicious entity creates a large number of fake identities (nodes) to gain disproportionate influence over the network, disrupt its consensus, and potentially launch other attacks'
    'justification': 'It implies a desirable goal that motivate action, such as pursuing social order, justice, and helpfulness, because they represent what an individual finds important.'
    'sense': 'an advantage to the benefit of someone or something'
    'justification': 'It describes a general principle that is relevant across diverse actions and situations, distinguishing it from more context-specific norms and attitudes.'
    'sense': {input_sense}
    'justification':
    """
    output_type = HumanValueJustificationOutput
    #print(prompt_str)
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result


def prompt_predict_opposite_term(model_name, input_term):
    prompt_str = f"""
        Identify the opposite, or antonym, of the input term. Follow the example.

        term: power
        opposite: powerlessness
        term: {input_term}
        opposite: 
    """
    output_type = OppositeTerm
    rag_agent = setup_agent(model_name, prompt_str, output_type)
    result = rag_agent.run_sync(prompt_str)
    return result


def prompt_predict_opposite_senses(rag_agent, input_term, input_list_senses):
    user_prompt_str = f"""
        Identify the opposite, or antonym, of the input term. Follow the example.

        term: power
        opposite: powerlessness
        senses: [the quality of lacking strength or power', 'being weak and feeble']
        term: {input_term}
        senses: {input_list_senses}
        opposite: 
    """
    output_type = OppositeSense


############################

def main_predict_human_value_sense_for_wordlist(in_file_schwarts_values, out_file_path_wordnet_senses, out_file_path_nonwordnet_senses):
    model_name = 'qwen3:8b'

    rag_agent1 = setup_agent_predict_wordnet_based_hv_senses(model_name)
    rag_agent2 = setup_agent_predict_misc_hv_senses(model_name)

    for term in read_lines_from_file(in_file_schwarts_values, lines_to_read = 30):
        wordnet_senses = get_wordnet_glosses(term.strip())
        print(wordnet_senses)
        if len(wordnet_senses) > 0:
            result = prompt_predict_wordnet_based_hv_senses(rag_agent1, term, wordnet_senses)
            json_output = result.output.model_dump_json()
            append_to_file(json_output, out_file_path_wordnet_senses)
        else:
            result = prompt_predict_misc_hv_senses(rag_agent2, term)
            json_output = result.output.model_dump_json()
            append_to_file(json_output, out_file_path_nonwordnet_senses)

def main_predict_non_value_sense_for_wordlist(in_file_path, out_file_path1, out_file_path2, num_lines=10, start_position=1):
    model_name = 'qwen3:8b'
    for term in read_lines_from_file(in_file_path, lines_to_read=num_lines, start_line_number=start_position):
        wordnet_senses = get_wordnet_glosses(term.strip())
        if len(wordnet_senses) > 0:
            result = prompt_predict_wordnet_based_nonvalue_senses(model_name, term.strip(), wordnet_senses)
            json_output = result.output.model_dump_json()
            append_to_file(json_output, out_file_path1)
        else:
            result = prompt_predict_misc_nonvalue_senses(model_name, term)
            json_output = result.output.model_dump_json()
            append_to_file(json_output, out_file_path2)

def main_predict_human_value_sense_for_word(term, out_file_path_wordnet_terms, out_file_path_unknown_terms):
    model_name = 'qwen3:8b'

    rag_agent1 = setup_agent_predict_wordnet_based_hv_senses(model_name)
    rag_agent2 = setup_agent_predict_misc_hv_senses(model_name)

    wordnet_senses = get_wordnet_glosses(term.strip())
    print(wordnet_senses)
    if len(wordnet_senses) > 0:
        result = prompt_predict_wordnet_based_hv_senses(rag_agent1, term, wordnet_senses)
        json_output = result.output.model_dump_json()
        append_to_file(json_output, out_file_path_wordnet_terms)
    else:
        result = prompt_predict_misc_hv_senses(rag_agent2, term)
        json_output = result.output.model_dump_json()
        append_to_file(json_output, out_file_path_unknown_terms)

def main_predict_noun_senses_in_sentences(num_lines, start_line, in_file_path, out_file_path):
    model_name = 'qwen3:8b'

    for sentence in read_lines_from_file(in_file_path, lines_to_read = num_lines, start_line_number=start_line):
        noun_senses = prompt_infer_noun_adjective_sense(model_name, sentence)
        json_output = noun_senses.output.model_dump_json()
        append_to_file(json_output, out_file_path)

def main_predict_compound_senses_in_sentences(num_lines, start_line, in_file_path, out_file_path):
    model_name = 'qwen3:8b'

    for sentence in read_lines_from_file(in_file_path, lines_to_read = num_lines, start_line_number=start_line):
        noun_senses = prompt_infer_compund_sense(model_name, sentence)
        json_output = noun_senses.output.model_dump_json()
        append_to_file(json_output, out_file_path)

def main_predict_opposite_term(num_lines, start_line, out_file_path):
    in_file_path = 'unique_values_google.txt'
    model_name = 'qwen3:8b'

    for term in read_lines_from_file(in_file_path, lines_to_read = num_lines, start_line_number=start_line):
        print("...", term)
        opposite_term = prompt_predict_opposite_term(model_name, term)
        json_output = json.loads(opposite_term.output.model_dump_json())
        with open(out_file_path, 'a') as f:
            f.write(json_output['term'].strip())
            f.write("\n")

def main_infer_why_human_value_sense(sense):
    model_name = 'qwen3:8b'
    output = prompt_infer_human_value(model_name, sense)
    return output


