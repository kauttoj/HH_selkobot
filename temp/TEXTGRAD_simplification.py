import textgrad as tg
from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
from textgrad.tasks import load_task
from textgrad.loss import TextLoss

model_string = 'gemma2_27BQ4'
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_engine = ChatExternalClient(client=client, model_string=model_string)

evaluation_instruction = tg.Variable("Is ths a good joke?",role_description="question to the LLM",  requires_grad=False)
eval_fn = TextLoss(evaluation_instruction,llm_engine)

response = tg.Variable("What did the fish say when it hit the wall? Dam.",role_description="Response from LLM", requires_grad=True)
loss = eval_fn(response)
loss.backward()



tg.set_backward_engine(llm_engine)

_, val_set, _, eval_fn = load_task("BBH_word_sorting", llm_engine)
question_str, answer_str = val_set[0]
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)

system_prompt = tg.Variable("You are a concise LLM. Think step by step.",
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))

prediction = model(question)

loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))

loss.backward()

optimizer.step()

prediction = model(question)