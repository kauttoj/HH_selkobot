import os
import numpy as np
from numpy.linalg import norm
import tiktoken
from collections import Counter
import sacrebleu
import sacremoses
from packaging import version
import torch.nn.functional as F
import re
from openai import OpenAI
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import spacy
from opik.evaluation.metrics import GEval
from dotenv import load_dotenv
import json

load_dotenv('.env')

# Spacy init
# python -m spacy download fi_core_news_lg
spacy_nlp = spacy.load('fi_core_news_lg') # for tokenizing and lemmatizing

# E5 init
E5_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL_NAME)
e5_model = AutoModel.from_pretrained(E5_MODEL_NAME, trust_remote_code=True)

# Jina init
JINA_MODEL_NAME = "jinaai/jina-embeddings-v3"
jina_model = AutoModel.from_pretrained(JINA_MODEL_NAME, trust_remote_code=True)

# openAI init
gpt4_config = {
    "temperature": 0.05,
    "model": 'gpt-4o', # 'gpt-4o-mini',
    "timeout": 60
}
OPENAI_EMBED_MODEL = "text-embedding-3-large"
tiktoken_enc = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)

openai_client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

### SARI function definitions
def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        # Fix an alleged bug [2] in the keep score computation.
        # keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    #      a target exactly.
    keepscore_precision = 1
    keepscore_recall = 1
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        # Fix an alleged bug [2] in the keep score computation.
        # keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    delscore_precision = 1
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1

    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    addscore_precision = 1
    addscore_recall = 1
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)

def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.split(" ")
    c1grams = csent.split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)
    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    return finalscore

def normalize(sentence, lowercase: bool = True, tokenizer: str = "spacy", return_str: bool = True,lemmatize: bool = False):

    # Normalization is requried for the ASSET dataset (one of the primary
    # datasets in sentence simplification) to allow using space
    # to split the sentence. Even though Wiki-Auto and TURK datasets,
    # do not require normalization, we do it for consistency.
    # Code adapted from the EASSE library [1] written by the authors of the ASSET dataset.
    # [1] https://github.com/feralvam/easse/blob/580bba7e1378fc8289c663f864e0487188fe8067/easse/utils/preprocessing.py#L7

    if len(sentence)==0:
        if not return_str:
            return [sentence]
        else:
            return sentence

    if tokenizer == 'spacy':
        doc = spacy_nlp(sentence)

        allowed_chars = r'^[a-zA-Z0-9.,!?\'"()-]+$'
        if lemmatize:
            normalized_sent = [str(token.lemma_) for token in doc if bool(re.match(allowed_chars, str(token.lemma_)))]
        else:
            normalized_sent = [str(token) for token in doc if bool(re.match(allowed_chars, str(token)))]
        normalized_sent = " ".join([x.lower() if lowercase else x for x in normalized_sent])
    else:
        if tokenizer in ["13a", "intl"]:
            if version.parse(sacrebleu.__version__).major >= 2:
                normalized_sent = sacrebleu.metrics.bleu._get_tokenizer(tokenizer)()(sentence)
            else:
                normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
        elif tokenizer == "moses":
            normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
        elif tokenizer == "penn":
            normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
        else:
            normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent

def SARI_compute(sources, predictions, references,lemmatize):
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Sources length must match predictions and references lengths.")
    sari_score = 0
    for src, pred, refs in zip(sources, predictions, references):
        sari_score += SARIsent(normalize(src,lemmatize=lemmatize), normalize(pred,lemmatize=lemmatize), [normalize(sent,lemmatize=lemmatize) for sent in refs])
    sari_score = sari_score / len(predictions)
    return sari_score

def get_sari_score(source,prediction,reference,lemmatize=False):
    sari_score = SARI_compute([source],[prediction],[[reference]],lemmatize)
    return sari_score

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return

def get_openai_scores(model_prediction, golden_target):
    SYSTEM_PROMPT = '''
    You are simulating the perspective of the target news consumer for simplified Finnish texts ("selkosuomi"). Selkosuomi is a form of the Finnish language that uses simpler vocabulary, grammar, and sentence structures to make content more accessible to people with limited language proficiency, such as immigrants learning Finnish. It aims to present information clearly and straightforwardly, avoiding complex expressions and idioms.
    Typical readers are immigrants between 20-50 years old who are living in Finland and are learning Finnish. They have limited Finnish language proficiency and rely on simplified texts to understand news articles.

    Your task is to compare two "selkosuomi" texts written in Finnish: an Input Text and a Reference Text. You provide a numerical score 0-100 corresponding how close the Input Text is to Reference Text.

    Instructions:
    Adopt the Target Audience Perspective. Evaluate the texts as a reader who is an immigrant aged 20-40, living in Finland and learning Finnish language.
    Consider how understandable, engaging, and culturally relevant the texts are for someone with your background.
    Pay attention to cultural references, idioms, or contexts that may or may not be clear to the target audience.

    Understand "Selkosuomi":

    1 Vocabulary
    -Uses common, everyday words.
    -Avoids jargon, technical terms, and rare words.
    -Explains necessary difficult words in simple terms.

    2 Grammar and Sentence Structure
    -Uses simple sentence structures (e.g., subject-verb-object).
    -Avoids complex clauses and passive voice.
    -Keeps sentences short and to the point.

    3 Style and Formatting
    -Uses clear and direct language.
    -Organizes content logically with clear headings and paragraphs.
    -May use lists or bullet points for clarity.

    4 Content
    -Focuses on essential information.
    -Avoids unnecessary details and digressions.
    -Presents information in a factual and neutral tone.

    Assess Similarity (Content and Meaning):

    1 Completeness
    -Check if the Input Text includes all the key information present in the Reference Text.
    -Identify any important details that are missing or any additional irrelevant information.

    2 Accuracy
    -Ensure that the facts presented in the Input Text are accurate and match those in the Reference Text.
    -Be alert to any misrepresentations or distortions of information.

    3 Consistency
    -Verify that the Input Text conveys the same overall message and intent as the Reference Text.
    -Assess whether the tone and emphasis are appropriate and consistent.

    Evaluate Readability and Simplicity:

    1 Language Level
    -Determine if the vocabulary is appropriate for someone learning Finnish.
    -Identify any complex words or phrases that could hinder understanding.

    2 Sentence Structure:
    -Check for the use of simple sentences.
    -Note any long or complex sentences that could be confusing.

    3 Clarity:
    -Assess whether ideas are expressed clearly and unambiguously.
    -Look for any confusing or vague statements.

    4 Flow and Cohesion:
    -Ensure that sentences and paragraphs flow logically.
    -Check for the use of connecting words to guide the reader.

    Identify Impact of Differences:

    1 Readability Impact
    -Analyze how any differences between the texts affect the ease of reading.
    -Determine if the changes improve or hinder comprehension.

    2 Content Impact
    -Evaluate whether omissions or additions change the meaning or importance of the information.
    -Assess the significance of any altered facts or messages.

    3 Cultural Relevance
    -Consider if any differences affect the cultural appropriateness or relevance to the target audience.
    -Identify any references that may not be understood by immigrants learning Finnish.

    Consider the Overall Effect:

    1 Engagement
    -Determine if the Input Text is engaging and holds the reader's interest.
    -Assess whether the text encourages the reader to continue reading.

    2 Suitability
    -Judge whether the Input Text is suitable for the intended audience in terms of content and language level.

    3 Effectiveness
    -Evaluate how effectively the Input Text communicates the intended message.

    # Scoring Guidelines #

    Explanation and Rationale:

    1 Provide a concise explanation of your evaluation, covering the key points from the instructions above.
    2 Highlight specific aspects that influenced your scoring decision.
    3 Keep the explanation clear and focused on the most important factors.

    Assign a Numerical Similarity Score between 0 and 100:

    100: The texts are identical in content, meaning, readability, and quality.
    90-99: Differences are minimal and do not affect readability, quality, or factual content. Any differences are insignificant and do not hinder understanding.
    80-89: Minor differences that slightly impact readability or content, but the overall message remains clear. May include minor omissions or slightly complex wording.
    70-79: Noticeable differences that somewhat affect readability and understanding. Some important content may be altered or missing, or language may occasionally be too complex.
    60-69: Clear differences that affect comprehension. Multiple instances of missing information or complex language that could confuse the reader.
    50-59: Significant differences leading to confusion. Important details are incorrect, omitted, or overshadowed by complex language. Readability is substantially hindered.
    30-49: Major differences with substantial impact on understanding. The Input Text fails to convey much of the essential information and is difficult to read.
    10-29: The texts are substantially different. The Input Text is poor in quality, contains numerous differences, and does not convey the same message.
    1-9: The Input Text bears minimal resemblance to the Reference Text. It is largely different topic or have very different content.
    0: The texts are completely different with no similarity in content or quality.

    # Output Format #

    Explanation: Provide a concise explanation of your evaluation and the rationale behind your score, focusing on the most impactful differences

    Score: On a new line, provide only the NUMERICAL SCORE between 0 and 100.

    DO NOT include any additional comments beyond the explanation and the score in your response.

    '''

    USER_INPUT = '''Texts to Compare (in Finnish)

    # INPUT TEXT #

    {input}

    # REFERENCE TEXT # 

    {reference}

    # OUTPUT #

    Your response:
    '''

    scores = []
    for iteration in range(3):
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_INPUT.format(input=model_prediction, reference=golden_target),
                }
            ],
            **gpt4_config
        )
        resp = response.choices[0].message.content.strip()
        ind = resp.find('Score:')
        if ind > -1:
            gpt4_score = float(resp[ind + len('Score:'):].strip())
            assert 0 <= gpt4_score <= 100, 'Score outside 0-100 limits!'
            gpt4_score = gpt4_score
        else:
            raise Exception('OpenAI did not return score!')

        scores.append(gpt4_score)

    gpt4_score = np.mean(scores)

    def processor(t_in):
        t_out = t_in
        while 1:
            res = tiktoken_enc.encode(t_out)
            if len(res) > 8192 * 0.95:
                t_out = t_out[0:-1]
            else:
                break

        if len(t_in) > len(t_out):
            print(f'cutted input by {len(t_in) - len(t_out)}')
        return t_out

    batch_data = [processor(x) for x in [model_prediction, golden_target]]
    # Process the batch using the client
    restp = openai_client.embeddings.create(input=batch_data, model=OPENAI_EMBED_MODEL, timeout=30)

    # Append the processed batch to the output list
    out = [np.array(x.embedding) for x in restp.data]
    embed_score = np.dot(out[0], out[1]) / (norm(out[0]) * norm(out[1]))

    return embed_score, gpt4_score

def get_e5_similarity(model_prediction,golden_target):
    # Each query must come with a one-sentence instruction that describes the task.
    # The task definition should be a one-sentence instruction that describes the task. This is a way to customize text embeddings for different scenarios through natural language instructions.
    task = 'Compare similarity of the machine-generated simplified Finnish text against a human-written gold-label reference text.'
    queries = [
        f'Instruct: {task}\nQuery: {golden_target}'
    ]
    # No need to add instruction for retrieval documents
    input_texts = queries + [model_prediction]

    # Tokenize the input texts
    batch_dict = e5_tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = e5_model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    score = float((embeddings[0] @ embeddings[1].T))

    return score

def get_jina_similarity(model_prediction,golden_target):
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
    embeddings = jina_model.encode([golden_target,model_prediction], task="text-matching")
    score = embeddings[0] @ embeddings[1].T
    return score

def get_geval_score(model_prediction,golden_target):
    metric = GEval(
        task_introduction="You are an expert linguist and judge tasked with evaluating the similarity of given two Finnish texts based on their content, readability, facts and typography.",
        evaluation_criteria="Compare OUTPUT text against REFERENCE text. Check for similarity of facts, writing style, text structure, vocabulary, complexity and readability. In the ideal case, OUTPUT and REFERENCE are identical letter-by-letter.",
        model='gpt-4o-mini'
    )

    res = metric.score(
        input={"OUTPUT": model_prediction, "REFERENCE": [golden_target]}
    )
    return res.value

def get_selkomittari_score(model_prediction):
    with open(os.getcwd() + r'\prompts\selkomittari_evaluator.txt','r',encoding='utf-8') as f:
        evaluator_system_prompt = f.read()

    scores = []
    for iteration in range(3):
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": evaluator_system_prompt},
                {
                    "role": "user",
                    "content": 'Arvio seuraava teksti:\n\n' + model_prediction,
                }
            ],
            **gpt4_config
        )
        resp = response.choices[0].message.content.strip()
        ind = resp.find('Pisteytykseni perusteluineen:')
        if ind<0:
            raise(Exception('incorrect output'))
        resp1 = resp[(ind+len('Pisteytykseni perusteluineen:')):].strip()
        try:
            scores = json.loads(resp1)
        except:
            raise(Exception('failed to parse JSON string: %s' % resp1))

        subsection_scores = [
            {'items':[5,6,7]},
            {'items': [8]},
            {'items': [9,10,11,12,13]},
            {'items': [14,15]},
            {'items': [21,22,23,24,25]},
            {'items': [26,27,28,29]},
            {'items': list(range(35,41))},
            {'items': list(range(41,49))},
            {'items': list(range(49, 54))}
        ]
        for k in range(0,len(subsection_scores)):
            subsection_scores[k]['scores']=[]

        main_score = 0
        main_count = 0
        apu_score = 0
        apu_count = 0
        scoremap = {'2':3.2,'1':0.8,2:3.2,1:0.8}
        for x in scores:
            num = int(x['kriteeri'])
            if x['pisteet'] > 0:
                if num in [1,2,3,4,16,17,18,19,20,30,31,32,33,34]:
                    main_count += 1
                    main_score += float(int(x['pisteet']))
                else:
                    for k in range(0,len(subsection_scores)):
                        if num in subsection_scores[k]['items']:
                            subsection_scores[k]['scores'].append(scoremap[x['pisteet']])
                            break

        for k in range(0,len(subsection_scores)):
            if len(subsection_scores[k]['scores'])>0:
                apu_count += 1
                apu_score += np.round(np.mean(subsection_scores[k]['scores']))

        if main_score==0:
            return 0

        score = (main_score + apu_score) / (main_count + apu_count)
        return score

#################################################################

if 0:
    original_text = '''Vantaan kaupunki luopuu päiväkodeissa järjestettävistä valokuvauksista. Kaupunki tiedotti huoltajia asiasta tällä viikolla. Muutos tulee voimaan ensi vuoden toukokuussa. Tuolloin päättyy voimassa oleva hankintakausi. On siis mahdollista, että joissain vantaalaisissa päiväkodeissa järjestetään valokuvaus vielä kuluvan syksyn tai kevään aikana.
    Kaikki perheet eivät halua lastensa osallistuvan kuvauksiin. Vantaan kaupungin varhaiskasvatusjohtaja Mikko Mäkelä perustelee huoltajille lähetetyssä kirjeessä päätöstä seuraavasti: – Valokuvauksilla on pitkät perinteet, mutta niiden järjestäminen on osoittautunut käytännössä haastavaksi. Valokuvauksille ei ole pedagogisia perusteita, eivätkä ne ole varhaiskasvatussuunnitelman mukaista toimintaa, Mäkelä toteaa.
    Koulukuvauksella on pitkät perinteet Vantaalla – kuvien historia ulottuu yli sadan vuoden taakse
    Vantaan kunnalliset päiväkodit ovat saaneet tähän saakka itse päättää, järjestetäänkö niissä vuosittainen valokuvaus vai ei. Viestissään kaupunki perusteleekin tehtyä päätöstä yhdenvertaisuudella.
    Vantaan Sanomien tietojen mukaan osalle perheistä vuosittainen valokuvaus on mieluinen ja odotettu asia, ja tuore päätös tuntuu harmilliselta. Kaupungin huoltajille lähetyn viestin mukaan "kaikki perheet eivät halua lastensa osallistuvan kuvauksiin". Toimintakaudesta 2025–2026 yhteinen linjaus siis on, että kuvausta ei järjestetä.
    '''
    model_prediction='''Vantaan kaupunki ei enää järjestä valokuvauksia päiväkodeissa. Kaupunki kertoi asiasta vanhemmille tällä viikolla. Muutos alkaa toukokuussa ensi vuonna. Silloin päättyy nykyinen sopimus. Joissakin päiväkodeissa voi olla vielä valokuvaus tänä syksynä tai keväänä.
    Kaikki perheet eivät halua, että heidän lapsensa osallistuvat valokuvaukseen. Vantaan varhaiskasvatusjohtaja Mikko Mäkelä selittää päätöstä kirjeessä vanhemmille. Hän sanoo, että valokuvauksilla on pitkät perinteet, mutta niiden järjestäminen on vaikeaa. Valokuvauksille ei ole opetuksellisia syitä, eikä se kuulu varhaiskasvatuksen suunnitelmaan.
    Vantaalla on ollut koulukuvauksia yli sata vuotta. Aiemmin päiväkodit saivat itse päättää, järjestetäänkö valokuvaus. Kaupunki perustelee päätöstä sillä, että kaikilla päiväkodeilla on nyt samat säännöt.
    Vantaan Sanomat kertoo, että osa perheistä pitää valokuvausta mukavana ja odotettuna asiana. Päätös voi tuntua heistä harmilliselta. Kaupunki sanoo viestissään, että "kaikki perheet eivät halua lastensa osallistuvan kuvauksiin". Vuodesta 2025–2026 alkaen valokuvausta ei enää järjestetä.'''

    golden_target='Vantaan kaupunki aikoo lopettaa päiväkotien valokuvaukset. Muutos tulee voimaan toukokuussa. Silloin päättyy voimassa oleva hankintakausi. Joissain vantaalaisissa päiväkodeissa saattaa olla valokuvaus vielä tänä syksynä tai ensi keväänä. Vantaan kaupungin varhaiskasvatuksen johtaja Mikko Mäkelä kertoo vanhemmille lähetetyssä kirjeessä, miksi päätös tehtiin. Hän sanoo huoltajille kirjeessä näin: – Valokuvauksilla on pitkät perinteet, mutta niiden järjestäminen on osoittautunut käytännössä haastavaksi. Valokuvauksille ei ole pedagogisia perusteita, eivätkä ne ole varhaiskasvatussuunnitelman mukaista toimintaa, Mäkelä sanoo. Tähän asti Vantaan kunnalliset päiväkodit ovat saaneet päättää itse, järjestetäänkö valokuvaus vai ei. Kirjeessään kaupunki perustelee, että päätös on yhdenvertainen. Osalle perheistä joka vuosi järjestetty valokuvaus on mukava ja odotettu asia, ja uusi päätös tuntuu harmilliselta. Kaupungin kirjeessä sanotaan, että "kaikki perheet eivät halua lastensa osallistuvan kuvauksiin". Toimintakautena 2025–2026 kuvausta ei järjestetä.'

    ### SARI ###

    sari_score = get_sari_score(original_text,model_prediction,golden_target,lemmatize=False)
    sari_score_lemma = get_sari_score(original_text,model_prediction,golden_target,lemmatize=True)
    print(f'SARI similarity scores: normal {sari_score}, lemmatized {sari_score_lemma}')

    ### e5 model ###

    e5_score = get_e5_similarity(model_prediction,golden_target)
    print(f'{E5_MODEL_NAME} similary score {e5_score }')

    ### JINA-AI ###


    # Compute similarities
    jina_score = get_jina_similarity(model_prediction,golden_target)

    print(f'{JINA_MODEL_NAME} similary score {jina_score}')

    ### OPENAI ###

    embed_score,gpt4_score = get_openai_scores(model_prediction,golden_target)

    print(f'OpenAI similary scores: GPT-4 {gpt4_score}, embedding {embed_score}')

    #### G-EVAL


    #from deepeval.metrics import GEval
    #from deepeval.test_case import LLMTestCase

    #metric = GEval()
    #test_case = LLMTestCase(
    #    input="What if these shoes don't fit?",
    #    actual_output="We offer a 30-day full refund at no extra costs.",
    #    retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    #)
    #metric.measure(test_case)
    #print(metric.score)

    print('\n\nall done!')

