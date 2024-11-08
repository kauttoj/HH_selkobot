import glob
from typing import Annotated
import json
import pandas as pd
import sqlite3
import os
import re
import autogen
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat
from openai import OpenAI
from anthropic import Anthropic

import time
from dotenv import load_dotenv

load_dotenv('.env')
# .env file is a text file that contains lines:
# OPENAI_API_KEY="[your key]"   <--- so far only this is needed
# ANTHROPIC_API_KEY="[your key]"
# GOOGLE_API_KEY="[your key]"

MAXIMUM_ROUNDS = 3 # how many rounds maximum of improvement, terminate early if not already finished. The last version of the text is returned.

gpt4_config_mini = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.05,
    "model": 'gpt-4o-mini',
    "timeout": 60,
}
gpt4_config_full = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.05,
    "model": 'gpt-4o',
    "timeout": 60
}
claude_config = {
    "temperature": 0.05,
    "model": "claude-3-5-sonnet-20241022",
    "timeout": 60,
    "api_type": "anthropic",
    "api_key":os.getenv('ANTHROPIC_API_KEY')
}

# prefill with this news text for fast testing purposes
DEFAULT_TEXT = '''<title>Nelilapsinen helsinkiläisperhe päätti luopua autosta – näin kävi</title>
<lead>Joukkuelajeja harrastava perhe on liikkunut nyt vuoden julkisilla.</lead>
Ilkka Taivassalon perhe luopui autosta vuosi sitten ja siirtyi käyttämään pääasiassa julkista liikennettä.
 <quote> Luovuimme autosta kesän alussa. Jossain kohtaa huomasin laihtuneeni ihan älyttömästi. Kun kävin kesän lopulla puntarilla, huomasin että painoni oli tippunut kymmenen kiloa kolmessa kuukaudessa, Taivassalo kertoo.</quote>
 Autottomuus oli tuonut päiviin paljon lisää arkiliikuntaa ja askelia. Muuta muutosta Taivassalo ei ollut päivittäiseen elämäänsä tehnyt.
 <quote> Söin ihan saman verran herkkuja ja pullaa, koska olen ihan jälkiruokafanaatikko. Laihtumiselle ei ollut mitään muuta selitystä kuin autosta luopuminen.</quote>
 Kiinteistöalalla päällikkönä työskentelevällä Taivassalolla ja hänen asiantuntijatehtävissä toimivalla puolisollaan on neljä lasta. Perhe asuu Munkkiniemessä.
 Lähipiirissä autotonta lapsiperhettä ei ole ihmetelty, mutta Taivassalon työkaverit kiinteistöalalla pitävät asiaa outona.
 <quote> Auton ajatellaan olevan statusasia kiinteistöasiantuntijan tehtävissä, Taivassalo sanoo.</quote>
 Taivassalo kertoo käyttäneensä autoa ennen "ihan kaikkeen". Hän kulki sillä harrastuksiin, töihin, ruokakauppaan, viikonloppureissuille ja kesälomareissuille. Myös huono keli oli syy käyttää autoa.
 Perhe käyttää nykyään yhteiskäyttöautoa pari kertaa kuukaudessa. Sillä saatetaan matkustaa esimerkiksi toiseen kaupunkiin. Ruuat perhe tilaa kotiinkuljetuksella.
 Viikoittaisiin harrastuksiin perhe kulkee julkisilla. Perheen yksi lapsi ja molemmat aikuiset harrastavat ringetteä. Toisen lapsen harrastus on jalkapallo. Perheellä on yhteensä 5–6 treenit viikossa. Siihen päälle on vielä pelit.
 <quote> Sunnuntaisin ringettetreeneihin menee 1,5 tuntia enemmän kuin autolla menisi. Kesällä menimme puolisoni kanssa häihin kolmella eri bussilla. Kesken matkan tajusimme, että meillä on väärän vyöhykkeen liput, Taivassalo kertoo.</quote>
 HSL:n eri vyöhykkeiden hahmottaminen on ollut Ilkka Taivassalolle vaikeaa. Hän pitää matkalippujen hintaa sopivana, mutta toivoo silti, että olemassa olisi pelkkä A-vyöhyke, jolla hän liikkuu suurimman osan ajasta.
 Mikä sitten sai perheen luopumaan autosta?
 <quote> Minulla oli töiden kautta leasing-auto, jonka sopimus oli päättymässä. Olin asian kanssa hieman viime tingassa liikkeellä, ja uuden auton saaminen olisikin kestänyt kuukausia. Päätimme, että eiköhän kokeilla olla ilman autoa.</quote>
 Perhe voisi ottaa joskus vielä auton eikä koe että asian suhteen tarvitsee olla "absolutisti".
 Julkisista kulkuvälineistä ratikka on isän ja lasten suosikki. Perheen vanhemmat lapset ovat 13-, 7- ja 4-vuotiaita. Nuorin lapsi Wilma on 10 kuukautta vanha, ja Taivassalo hoitaa häntä tällä hetkellä kotona.
 <quote> Lapset käyttäytyvät 90 % ajasta hyvin julkisissa liikennevälineissä. Joskus kun ei ole osattu käyttäytyä, niin olen noussut lasten kanssa kyydistä pois. Yritän elää niin, että minulla ei ole kiire mihinkään, Taivassalo sanoo.</quote>
 Vain kerran Taivassalo on todella toivonut, että hänellä olisi ollut auto.
 <quote> Oli marraskuu ja olin tulossa pitkältä kiinteistökierrokselta. Seisoin räntäsateessa litimärkänä Kehä 1:n varrella, kun kaksi bussia jätti välistä. Kun kolmas bussi lopulta tuli, niin aikaa oli tuhlaantunut jo puolitoista tuntia.</quote>
 Monella muullakin Helsingin Uutisten lukijalla on myönteisiä kokemuksia autottomuudesta, selvisi HU:n elokuun aikana tekemästä kyselystä. Vastauksia tuli 60. Valtaosa joukkoliikenteen käyttäjistä oli tyytyväisiä liikkumisen sujuvuuteen. Vastaajat pitivät erityisesti julkisen liikenteen hintoja kohtuullisina autoiluun verrattuna.
 <quote> Muutimme kirjaimellisesti metroaseman yläpuolelle päästäksemme autosta eroon. Perheelle on viikossa seitsemän tuntia enemmän aikaa, ja luonto kiittää. Ei kaduta, kertoo Redin tornitalon 25:een kerrokseen muuttanut nainen Kalasatamasta.</quote>
 Välillä muualla asunut Vilma Partanen pitää pääkaupunkiseudun julkista liikennettä edistyksellisenä muihin kaupunkeihin verrattuna.
 <quote> Paluumuutin Helsinkiin muutama vuosi sitten. Melkein ensimmäiseksi myin auton pois. HSL-liikennöinti on mielestäni huippua, eikä tulisi mieleenkään palata takaisin autoilijaksi. Monet mollaavat kallista hintaa, mutta kyllä auton omistaminen on monin kerroin kalliimpaa, Partanen sanoo kyselyvastauksessaan.</quote>
 Antti Seppälä kertoo kyselyssä luopuneensa autosta, kun muutti muutama vuosi sitten Espoosta Jätkäsaareen.
 <quote>Matkustan nyt julkisilla viisi kertaa viikossa Jätkäsaaresta lentoasemalle. Pääosin matka sujuu mukavasti äänikirjoja kuunnellen. En ole kaivannut omaa autoa päivääkään.</quote>
 Kyselyyn tuli myös päinvastaisia vastauksia, joissa julkisen liikenteen käyttäjä oli vaihtanut takaisin autoon.'''

factchecker_system_prompt = '''Faktantarkastaja. Olet faktantarkastaja, jonka tehtävä on tarkastaa, että uusi muunnettu teksti vastaa keskeisten FAKTOJEN osalta alkuperäistä tekstiä. Et ota kantaa tekstin luettavuuteen tai selkeyteen, ainoastaan asiavirheisiin. 
 Käyt yksityskohtaisesti kaikki tekstissä esiintyvät faktat läpi, listaat ne ja varmistat, että faktat eivät oleellisesti muutu tai vääristy uudessa tekstissä. Asiat voidaan kertoa eri tavalla ja eri muodoissa, mutta keskeisten faktojen on aina pysyttävä ennallaan

 Esimerkkejä keskeisistä faktoista:
 -lukumäärät ja numerot
 -erisnimet
 -tittelit ja ammatit
 -tapahtumat
 -väitteet
 -ajankohdat

 Faktoiksi EI lasketa kielellisiä ilmaisuja, tekstin muotoilua tai lausejärjestystä, jotka eivät vaikuta varsinaiseen asiasisältöön.
 
 Älä huomioi teksteissä olevia tageja
 <headline>...</headline>
 <lead>...</lead>
 <subtitle>...</subtitle>
 <quote>...</quote> 
 Ne eivät ole osa tarkastettavaa tekstiä, vaan niitä käytetään ainoastaan tekstin ladontaan.

 # Alkuperäinen teksti (lähdemateriaali) #

 {old_text}

 # Tehtävä #

 Vedä syvään henkeä ja lue alkuperäinen teksti (lähdemateriaali) ja annettu uusi teksti (kohdemateriaali) huolellisesti läpi. 
 Vertaile alkuperäisessä ja uudessa tekstissä kerrottuja faktoja toisiinsa. Älä keskity sanamuotoihin tai oikeinkirjoituksiin, ainoastaan faktoihin.
 Tämän jälkeen kirjoita auki havaintosi ja anna yksityiskohtainen palautteesi sekä tarvittavat korjaukset uuteen tekstiin.
 ET KOSKAAN KIRJOITA UUTTA VERSIOTA TEKSTISTÄ, ainoastaan ehdotat kriittisiä korjauksia uuden tekstin eri kohtiin, mikäli sellaisia on.    

 # Vastaus # 

 Arvioni tekstistä:
 [yhteenveto uuden tekstin faktojen paikkansapitävyydestä]

 Korjausehdotukset:
 [lista tarvittavista faktojen korjauksista uuteen tekstiin]
 '''

editor_system_prompt = """Editori. Olet uutismedian päätoimittaja ja kirjoitetun viestinnän kokenut huippuasiantuntija. Tehtävänäsi on päättää milloin annetun tekstin hienosäätäminen voidaan lopettaa ja teksti on valmis julkaistavaksi yleisölle.
Arvioit annettua tekstiä ja siihen ehdotettujen korjausten välttämättömyyttä, sekä tekstin julkaisukelpoisuutta. Luet alkuperäisen tekstin, uuden selkosuomenkielisen tekstin ja annat lyhyen arviosi ehdotetuista korjauksista. 
Sinun on arvioitava tarkasti tekstin laatua ja julkaisukelpoisuutta, mutta samalla huomioitava lisäkustannukset, jotka jokainen tekstin uudelleenkirjoitus ja analyysi maksavat. Punnitse erittäin tarkasti laatua ja kustannustehokkuutta. 

Arviosi jälkeen valitse toinen seuraavista vaihtoehdoista (nämä ovat ainoat vaihtoehdot joita sinulla on):
1. Jos teksti on jo riittävän hyvä ja korjausehdotukset eivät ole tarpeellisia, vastaa 'TERMINATE'
2. Jos korjaukset ovat tarpeellisia ja teksti on kirjoitettava uudelleen, vastaa 'Kirjoitetaan uusi versio tekstistä ja korjataan puutteet.'

Anna vastauksesi seuraavassa muodossa:

Arvioni tekstistä:
[yleinen arviosi tekstin julkaisukelpoisuudesta]

Päätökseni:
[valittu vaihtoehto]
"""

writer_initial_command = 'Lue annettu alkuperäinen uutisteksti tarkasti. Mukauta se selkosuomeksi selkosuomen kriteerien pohjalta.'
writer_rewrite_command = 'Lue annettu alkuperäinen uutisteksti, aikaisemmin kirjoittamasi selkosuomenkielinen teksti, sekä kaikki saamasi palaute ja korjausehdotukset tarkasti. Kirjoita saamasi palautteen pohjalta uusi ja parempi versio selkosuomeksi mukautetusta tekstistä.'

client = {'openai':
    OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    ),
        'claude':
    Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    ),
}

### bot definitions and prompts

SELKO_LOCATOR = 'Teksti selkosuomeksi:'

def filewriter(content,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
def filereader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def nullify_history(processed_messages):
    global latest_test
    processed_messages = [processed_messages[-1]] # take only the latest message as a reference
    if len(latest_text)==0:
        raise(Exception('Latest text is empty!'))
    processed_messages[0]['content'] = 'uusi teksti (kohdemateriaali):\n' + latest_text # nullify content
    return processed_messages

def process_all_messages_before_reply_hook(processed_messages):
    return processed_messages

def bot_constructor(writer_prompt=None,critic_prompt=None,factchecker_prompt=None,editor_prompt=None,llm=None):

    factchecker_prt = factchecker_prompt
    editor_prt = editor_prompt

    writer_prt = """Kirjoittaja. Olet selkosuomen huippuasiantuntija. Osaat kirjoittaa ja mukauttaa annettua suomenkielistä tekstiä helpompaan selkosuomen muotoon.
        
    {prompt}
    
    # Tehtävä #
    
    Kirjoita selkosuomeksi mukautettu versio annetusta yleiskielisestä uutistekstistä. 
    {instruction} 
    Saat muokata annettua yleiskielistä tekstiä, mutta ET SAA keksiä uutta sisältöä tai uusia faktoja, joita alkuperäisessä tekstissä ei kerrota.
    
    Erota tekstissäsi otsikko, ingressi, väliotsikot ja lainaukset aina seuraavilla tageilla, joita ovat:
     otsikko: <title>...</title>
     ingressi: <lead>...</lead>
     väliotsikko: <subtitle>...</subtitle>
     lainaukset: <quote>...</quote> 
    JOKAINEN teksti sisältää otsikon ja ingressin. Nämä eivät ole osa tekstiä, vaan niitä käytetään tekstin ladontaan verkkosivulle.
    
    # Vastaus #
    
    Anna vastauksesi täsmälleen seuraavassa muodossa
    
    Tarvittavat muutokset/korjaukset sanallisesti: 
    [lista tarvittavista muutoksista tekstiin]
    
    {selko_identifier}
    [teksti selkosuomeksi]
    """.replace('{prompt}',writer_prompt).replace('{selko_identifier}',SELKO_LOCATOR).replace('{instruction}',writer_initial_command)

    critic_prt = """Kriitikko. Olet selkosuomen huippuasiantuntija ja opettaja. Tunnet perinpohjaisesti selkosuomen kriteerit ja osaat arvioida tekstien selkeyttä, antaa rakentavaa palautetta ja korjausehdotuksia. 
    Tehtävänäsi on analysoida annettua tekstiä ja arvioida täyttääkö se selkosuomen keskeiset kriteerit. Ohjeistat kirjoittajaa tekstin parantamisessa ja puutteiden korjaamisessa.
    
    {prompt}
    
    Jokainen teksti sisältää aina otsikon ja ingressin eli tiivistelmäkappaleen. Lisäksi tekstissä voi olla väliotsikoita ja lainauksia. Kaikki nämä neljä kenttää on merkitty tageilla, joita ovat:
        otsikko: <title>...</title>
        ingressi: <lead>...</lead>
        väliotsikko: <subtitle>...</subtitle>
        lainaukset: <quote>...</quote>
    Nämä tagit eivät koskaan näy lukijalle tai ole osa tekstiä, vaan ne on lisätty selkeyden vuoksi.
    
    # Tehtävä #
    
    Arvioi annettua tekstiä selkosuomen periaatteiden pohjalta ja listaa kaikki välttämättömät korjaukset tekstiin. Anna ehdotuksesi aina seuraavassa muodossa. 
    
    # Vastaus #
    
    Anna vastauksesi täsmälleen seuraavassa muodossa
    
    Arvioni tekstistä:
    [yleinen arviosi tekstistä ja sen selkeydestä]
    
    Korjausehdotukseni:
    [yksityiskohtainen lista korjausehdotuksista tekstiin]
    """.replace('{prompt}',critic_prompt)

    def term_msg(msg):
        return "TERMINATE" in msg["content"]

    fact_checker = autogen.AssistantAgent( # autogen.AssistantAgent(
        name="Faktantarkastaja",
        llm_config=llm,
        description="Faktantarkastaja. Vertailee kahta annettua tekstiä toisiinsa ja tutkii ovatko ne faktojen osalta yhdenmukaiset.",
        is_termination_msg=None,#term_msg,
        system_message=factchecker_prt,
    )
    fact_checker.register_hook("process_all_messages_before_reply",nullify_history)

    writer = autogen.AssistantAgent(# autogen.AssistantAgent(
        name="Kirjoittaja",
        llm_config=llm,
        description="Kirjoittaja ja selkosuomen huippuasiantuntija ja toimittaja. Osaa muokata annettua suomenkielistä tekstiä helpommin luettavaan muotoon eli selkosuomentaa.",
        is_termination_msg=None,#term_msg,
        system_message=writer_prt,
    )
    writer.register_hook('process_message_before_send',post_message_processor)

    critic = autogen.AssistantAgent(
        name="Kriitikko",
        llm_config=llm,
        description="Kriitikko. Selkosuomen huippuasiantuntija ja opettaja. Osaa arvioida tekstin selkeyttä ja antaa tarvittaessa rakentavaa palautetta ja korjausehdotuksia tekstiin.",
        system_message=critic_prt,
        is_termination_msg=None, #lambda msg: "TERMINATE" in msg["content"],
    )
    editor = autogen.AssistantAgent(
        name="Editori",
        llm_config=llm,
        description="Editori. Uutismedian päätoimittaja ja kirjoitetun viestinnän kokenut huippuasiantuntija, joka lukee selkosuomeksi kirjoitetun tekstin ja arvioijien kommentit ja korjausehdotukset. Päättää voidaanko teksti julkaista vai pitääkö teksti kirjoittaa uudelleen korjattuna.",
        system_message=editor_prt,
        is_termination_msg=term_msg,
    )

    #evaluator = autogen.AssistantAgent(
    #    name="Evaluaattori",
    #    llm_config=llm,
    #    description="Evaluaattori. Selkosuomen ja viestinnän huippuasiantuntija, joka pisteyttää annetun tekstin selkomittaria käyttäen.",
    #    system_message=evaluator_prt,
    #    is_termination_msg=None,
    #)

    return {#"'evaluator':evaluator,
            'editor':editor,
            'critic':critic,
            'writer':writer,
            'fact_checker':fact_checker}

def post_message_processor(sender, message, recipient, silent):
    global latest_text
    global writing_loop
    identifier = SELKO_LOCATOR
    if sender.name == 'Kirjoittaja':
        ind = message.find(identifier)
        if ind > -1:
            latest_text = message[(ind + len(identifier)):].strip()
            if writing_loop==1:
                current_prompt = sender.system_message
                assert writer_initial_command in current_prompt
                sender.update_system_message(current_prompt.replace(writer_initial_command,writer_rewrite_command))

            # current_prompt = fact_checker.system_message
            # fact_checker.update_system_message(current_prompt.replace('{new_text}', latest_text))
        else:
            raise(Exception('\n!!!!!! No new text provided by writer !!!!!!!!\n'))
        writing_loop += 1
    return message

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

# Function to process input and display output
def process_text(input_text,agents_list,llm_config): #,output_box, console_box):
    global latest_text
    global writing_loop

    if len(input_text)<100:
        print('too short input text!')
        return

    latest_text = ''
    writing_loop = 1

    with open("logs.db", "w") as f:
        f.write("")
    logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
    print("Started Logging session ID: " + str(logging_session_id))

    #fact_checker.update_system_message(factchecker_system_prompt.replace('{old_text}', input_text))

    groupchat = autogen.GroupChat(
        agents=agents_list, messages=[], allow_repeat_speaker=False, max_round=MAXIMUM_ROUNDS*len(agents_list),
        select_speaker_message_template='You are in a role play game. The following roles are available: {roles}. Read the following conversation. Then select the next role from {agentlist} to play. Only return the role.',
        send_introductions=True, speaker_selection_method='round_robin'
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    user_proxy.initiate_chat(
        manager,
        message=f"Mukauta seuraava teksti selkosuomeksi:\n\n{input_text}",
    )
    # Stop logging
    autogen.runtime_logging.stop()

    # create function to get log
    def get_log(dbname="logs.db", table="chat_completions"):
        con = sqlite3.connect(dbname)
        query = f"SELECT request, response, cost, start_time, end_time from {table}"
        cursor = con.execute(query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        data = [dict(zip(column_names, row)) for row in rows]
        con.close()
        return data

    def str_to_dict(s):
        return json.loads(s)

    # use pandas to get extra information and print out to terminal
    def get_log(dbname="logs.db", table="chat_completions"):
        import sqlite3
        con = sqlite3.connect(dbname)
        query = f"SELECT request, response, cost, start_time, end_time from {table}"
        cursor = con.execute(query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        data = [dict(zip(column_names, row)) for row in rows]
        con.close()
        return data

    def str_to_dict(s):
        return json.loads(s)

    log_data = get_log()
    log_data_df = pd.DataFrame(log_data)

    log_data_df["total_tokens"] = log_data_df.apply(
        lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
    )

    log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"],axis=1)

    log_data_df["response"] = log_data_df.apply(
        lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
    )

    console_text = ''
    final_output = None
    for k,row in enumerate(log_data_df.iterrows()):
        console_text += f'\n\n-------------------------- iteration {k+1}----------------------------\n\n'
        role = row[1]['request'].split('.')[0]
        console_text += 'Current speaker: ' + role
        response = '\nResponse: [no response]' if row[1]['response'] is None else '\nResponse:\n\n' + row[1]['response']
        if 'kirjoittaja' in role.lower():
            ind = response.find(SELKO_LOCATOR)
            if ind>-1:
                final_output = response[(ind+len(SELKO_LOCATOR)):].strip()
        console_text += response

    output_text = latest_text

    print('Processing finished')

    return [output_text,console_text]

def get_llm_response(prompt,input_text,llm):
    if 'claude' in llm['model']:
        response = client['claude'].messages.create(
            model=llm['model'],
            max_tokens=6000,
            system=prompt,  # <-- role prompt
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        resp = response.content[0].text
    else:
        response = client['openai'].chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=llm['model'],
            temperature=llm['temperature'],
        )
        resp = response.choices[0].message.content
    return resp


def remove_tags(input_text):
    # Define various quotation marks that might appear
    quotation_marks = ['“', '”','«', '»','‘', '’','❝', '❞','〝', '〞']
    # Function to handle <quote> tags specifically
    def handle_quote(match):
        content = match.group(1).strip()
        # Remove any existing quotation marks
        if any([content[0] in quotation_marks]):
            content = content[1:]
        if any([content[-1] in quotation_marks]):
            content = content[0:-1]
        # Check if the content does not start and end with ASCII double quotes
        content = f'"{content}"'
        return content

    # Handle <quote> tags separately, adding ASCII double quotes if needed
    output_text = re.sub(r'<quote>(.*?)</quote>', handle_quote, input_text)

    # Remove all other tags
    output_text = re.sub(r'<[^>]+>(.*?)</[^>]+>', r'\1', output_text)

    return output_text

INPUT_PATH = os.getcwd() + os.sep + r'data\final' + os.sep
files = glob.glob(INPUT_PATH + '*.txt')
ids = sorted(list(set([x.split(os.sep)[-1].split('_selko.txt')[0] for x in files if '_selko.txt' in x])))
input_files = []
for id in ids:
    selko = INPUT_PATH + '%s_selko.txt' % id
    regular = INPUT_PATH + '%s_regular.txt' % id
    input_files.append({'selko':selko,'regular':regular,'id':id})

EVAL_FILE = 'evaluation_results_part1.pickle'
try:
    evaluation_results = pd.read_pickle(EVAL_FILE)
    print('continuing evaluation, %i items exists' % len(evaluation_results))
except:
    evaluation_results = pd.DataFrame(data={'id':[]})
    print('starting evaluation from scratch')

for loop in range(1):

    try:

        for llm in [gpt4_config_full]: #claude_config

            print('LLM = %s' % llm['model'])

            for file_num,f in enumerate(input_files[0:20]):

                input_text = filereader(f['regular'])
                reference_text = filereader(f['selko'])
                file_id = f['id']

                input_text_raw = remove_tags(input_text)
                reference_text_raw = remove_tags(reference_text)

                # Autogen bot - v1 full, no examples
                id = 'file%s_%s_(%s)' % (file_id,'Autogenbot_v1',llm['model'])
                if id not in evaluation_results['id'].to_list():

                    time.sleep(30)
                    writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v1.txt')
                    critic_system_prompt = writer_system_prompt
                    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)

                    bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                    writer = bots['writer']
                    critic = bots['critic']
                    factchecker = bots['fact_checker']
                    editor = bots['editor']

                    agents_list = [user_proxy, writer, critic, factchecker, editor]
                    output_text,_ = process_text(input_text,agents_list,llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # Autogen bot - v2 full, no examples

                id = 'file%s_%s_(%s)' % (file_id,'Autogenbot_v2',llm['model'])
                if id not in evaluation_results['id'].to_list():

                    time.sleep(30)
                    writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2.txt')
                    critic_system_prompt = writer_system_prompt
                    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)

                    bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                    writer = bots['writer']
                    critic = bots['critic']
                    factchecker = bots['fact_checker']
                    editor = bots['editor']

                    agents_list = [user_proxy, writer, critic, factchecker, editor]
                    output_text,_ = process_text(input_text,agents_list,llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # Autogen bot - v1 full, with examples
                #
                # id = '%s_(%s)' % ('Autogenbot_v1_examples',llm['model'])
                # if id not in evaluation_results['id']:
                #
                #     writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v1_examples.txt')
                #     critic_system_prompt = writer_system_prompt
                #     factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)
                #
                #     bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                #     writer = bots['writer']
                #     critic = bots['critic']
                #     factchecker = bots['fact_checker']
                #     editor = bots['editor']
                #
                #     agents_list = [user_proxy, writer, critic, factchecker, editor]
                #     output_text,_ = process_text(input_text,agents_list,llm)
                #
                #     evals = evaluate_result(output_text,reference_text,input_text)
                #     row = {'output_selko': output_text,'input_regular':f['regular'],'reference_selko':f['selko'],'model':llm['model'],'id':id} | evals
                #     evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                #     evaluation_results.to_pickle(EVAL_FILE)


                # Autogen bot - v2 full, with examples

                id = 'file%s_%s_(%s)' % (file_id,'Autogenbot_v2_examples',llm['model'])
                if id not in evaluation_results['id'].to_list():
                    time.sleep(30)
                    writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2_examples.txt')
                    critic_system_prompt = writer_system_prompt
                    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)

                    bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                    writer = bots['writer']
                    critic = bots['critic']
                    factchecker = bots['fact_checker']
                    editor = bots['editor']

                    agents_list = [user_proxy, writer, critic, factchecker, editor]
                    output_text,_ = process_text(input_text,agents_list,llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # Autogen bot - v1 small, no examples
                #
                # id = '%s_(%s)' % ('Autogenbot_v1_small',llm['model'])
                # if id not in evaluation_results['id']:
                #
                #     writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v1.txt')
                #     critic_system_prompt = writer_system_prompt
                #     factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)
                #
                #     bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                #     writer = bots['writer']
                #     critic = bots['critic']
                #     factchecker = bots['fact_checker']
                #     editor = bots['editor']
                #
                #     agents_list = [user_proxy, writer, factchecker, editor]
                #     output_text,_ = process_text(input_text,agents_list,llm)
                #
                #     evals = evaluate_result(output_text,reference_text,input_text)
                #     row = {'output_selko': output_text,'input_regular':f['regular'],'reference_selko':f['selko'],'model':llm['model'],'id':id} | evals
                #     evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                #     evaluation_results.to_pickle(EVAL_FILE)

                # Autogen bot - v2 small, no examples

                id = 'file%s_%s_(%s)' % (file_id,'Autogenbot_v2_small',llm['model'])
                if id not in evaluation_results['id'].to_list():
                    time.sleep(30)
                    writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2.txt')
                    critic_system_prompt = writer_system_prompt
                    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}',input_text)

                    bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt, factchecker_prompt=factchecker_system_prompt0,editor_prompt=editor_system_prompt, llm=llm)
                    writer = bots['writer']
                    critic = bots['critic']
                    factchecker = bots['fact_checker']
                    editor = bots['editor']

                    agents_list = [user_proxy, writer, factchecker, editor]
                    output_text,_ = process_text(input_text,agents_list,llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # Autogen bot - v1 small, with examples

                # Autogen bot - v1 small, no examples
                #
                # id = '%s_(%s)' % ('Autogenbot_v1_small_examples', llm['model'])
                # if id not in evaluation_results['id']:
                #     writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v1_examples.txt')
                #     critic_system_prompt = writer_system_prompt
                #     factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}', input_text)
                #
                #     bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt,
                #                            factchecker_prompt=factchecker_system_prompt0, editor_prompt=editor_system_prompt,
                #                            llm=llm)
                #     writer = bots['writer']
                #     critic = bots['critic']
                #     factchecker = bots['fact_checker']
                #     editor = bots['editor']
                #
                #     agents_list = [user_proxy, writer, factchecker, editor]
                #     output_text, _ = process_text(input_text, agents_list, llm)
                #
                #     evals = evaluate_result(output_text, reference_text, input_text)
                #     row = {'output_selko': output_text, 'input_regular': f['regular'], 'reference_selko': f['selko'],
                #            'model': llm['model'], 'id': id} | evals
                #     evaluation_results = pd.concat([evaluation_results, pd.DataFrame(row, index=[0])], ignore_index=True)
                #     evaluation_results.to_pickle(EVAL_FILE)
                #
                # # Autogen bot - v2 small, no examples

                id = 'file%s_%s_(%s)' % (file_id,'Autogenbot_v2_small_examples', llm['model'])
                if id not in evaluation_results['id'].to_list():
                    time.sleep(30)
                    writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2_examples.txt')
                    critic_system_prompt = writer_system_prompt
                    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}', input_text)

                    bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt,
                                           factchecker_prompt=factchecker_system_prompt0, editor_prompt=editor_system_prompt,
                                           llm=llm)
                    writer = bots['writer']
                    critic = bots['critic']
                    factchecker = bots['fact_checker']
                    editor = bots['editor']

                    agents_list = [user_proxy, writer, factchecker, editor]
                    output_text, _ = process_text(input_text, agents_list, llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # single call - type 1

                id = 'file%s_%s_(%s)' % (file_id,'single_bot_selkomedia_9rule', llm['model'])
                if id not in evaluation_results['id'].to_list():
                    prompt = '''
            Selkokieli on kielenkäytön muoto, joka on mukautettu erityisesti niille, joille yleiskieli on liian vaikeaa seurata. Selkokieltä käytetään kommunikoinnissa esimerkiksi erityisryhmille, joilla on vaikeuksia ymmärtää tai tuottaa yleiskieltä. Selkokieli on yksinkertaista, konkreettista, lauseet ovat lyhyitä ja asiayhteydet ilmaistaan selvästi.
            
            Selkokielen pääsäännöt:
            
            1. Yksinkertainen sanavalinta: Valitaan yksinkertaisia, tuttuja sanoja, jotka ovat mahdollisimman konkreettisia. Monimutkaiset tai harvinaiset sanat ja kieliopilliset rakenteet korvataan yksinkertaisemmilla.
            
            2. Lyhyet, yksinkertaiset lauseet: Yhdessä lauseessa kerrotaan yksi asia kerrallaan, ja lauseissa pyritään välttämään liitekyssymyksiä ja päälauseen sisällä olevia sivulauseita.
            
            3. Selkeä rakenne: Tekstin on oltava loogisesti jäsennelty ja siinä on pysyttävä asiassa.
            
            4. Konkreettisuus: Abstraktit käsitteet ja termit pyritään muuttamaan konkreettisiksi esimerkkien, kuvailujen tai määritelmien avulla.
            
            5. Aktiivilauseet: Passiivilauseet korvataan aktiivilauseilla. Esimerkiksi lause "läksyt on tehtävä" muutetaan muotoon "sinun täytyy tehdä läksyt".
            
            6. Monimuotoisuus: Selkokieli hyödyntää myös visuaalisia elementtejä, kuten kuvia ja symboleita, informaation selkeyttämiseksi.
            
            7. Toisto ja selventävä ilmaisu: Käytetään tarvittaessa toistoa ja selventäviä ilmauksia varmistamaan, että viesti on ymmärretty oikein.
            
            8. Suoria keskustelu tai lainauksia saa käyttää.
            
            9. Jos joudut käyttämään monimutkaisia sanoja, selitä mitä ne tarkoittavat.
            
            On hyvä muistaa, että selkokieli ei ole "helppoa" kieltä, vaan sen tavoitteena on tehdä informaation ymmärtäminen helpoksi niille, jotka tarvitsevat sitä. Selkokielen avulla pyritään saavuttamaan yhdenvertaisuutta ja osallisuutta.
            
            Muunna annettu teksti suoraan selkokielelle noudattaen edellä mainittuja ohjeita.
            Palaute pelkkä muunnettu teksti.
                    '''
                    time.sleep(30)
                    output_text = get_llm_response(prompt,input_text_raw,llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # single call - type 2

                id = 'file%s_%s_(%s)' % (file_id,'single_bot_selkomedia_simple', llm['model'])
                if id not in evaluation_results['id'].to_list():
                    prompt = '''Voisitko muuntaa artikkelissa näkyvät vaikeat sanat ja lauseet selkokielisemmäksi. Säilytä jutun rakenne ja sisältö mahdollisimman alkuperäisinä muilta osin.
                    Palauta pelkkä muunnettu teksti.'''

                    time.sleep(30)
                    output_text = get_llm_response(prompt, input_text_raw, llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                # single call - v1

                id = 'file%s_%s_(%s)' % (file_id,'single_bot_v2', llm['model'])
                if id not in evaluation_results['id'].to_list():
                    prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2.txt')
                    writer_prt = """Olet selkosuomen huippuasiantuntija. Osaat kirjoittaa ja mukauttaa annettua suomenkielistä tekstiä helpompaan selkosuomen muotoon.
            
                    {prompt}
            
                    # Tehtävä #
            
                    Kirjoita selkosuomeksi mukautettu versio annetusta yleiskielisestä uutistekstistä. 
                    {instruction} 
                    Saat muokata annettua yleiskielistä tekstiä, mutta ET SAA keksiä uutta sisältöä tai uusia faktoja, joita alkuperäisessä tekstissä ei kerrota.
            
                    Erota tekstissäsi otsikko, ingressi, väliotsikot ja lainaukset aina seuraavilla tageilla, joita ovat:
                     otsikko: <headline>...</headline>
                     ingressi: <lead>...</lead>
                     väliotsikko: <subtitle>...</subtitle>
                     lainaukset: <quote>...</quote> 
                    Nämä eivät ole osa tekstiä, vaan niitä käytetään tekstin ladontaan verkkosivulle.
            
                    # Vastaus #
            
                    Anna vastauksesi täsmälleen seuraavassa muodossa
            
                    Tarvittavat muutokset/korjaukset sanallisesti: 
                    [lista tarvittavista muutoksista tekstiin]
            
                    {selko_identifier}
                    [teksti selkosuomeksi]
                    """.replace('{prompt}', prompt).replace('{selko_identifier}', SELKO_LOCATOR).replace('{instruction}',writer_initial_command)

                    time.sleep(30)
                    output_text = get_llm_response(prompt, 'Muunna seuraava teksti selkosuomeksi:\n\n'+input_text, llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

                id = 'file%s_%s_(%s)' % (file_id,'single_bot_v2_examples', llm['model'])
                if id not in evaluation_results['id'].to_list():
                    prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2_examples.txt')
                    writer_prt = """Olet selkosuomen huippuasiantuntija. Osaat kirjoittaa ja mukauttaa annettua suomenkielistä tekstiä helpompaan selkosuomen muotoon.
        
                    {prompt}
        
                    # Tehtävä #
        
                    Kirjoita selkosuomeksi mukautettu versio annetusta yleiskielisestä uutistekstistä. 
                    {instruction} 
                    Saat muokata annettua yleiskielistä tekstiä, mutta ET SAA keksiä uutta sisältöä tai uusia faktoja, joita alkuperäisessä tekstissä ei kerrota.
        
                    Erota tekstissäsi otsikko, ingressi, väliotsikot ja lainaukset aina seuraavilla tageilla, joita ovat:
                     otsikko: <headline>...</headline>
                     ingressi: <lead>...</lead>
                     väliotsikko: <subtitle>...</subtitle>
                     lainaukset: <quote>...</quote> 
                    Nämä eivät ole osa tekstiä, vaan niitä käytetään tekstin ladontaan verkkosivulle.
        
                    # Vastaus #
        
                    Anna vastauksesi täsmälleen seuraavassa muodossa
        
                    Tarvittavat muutokset/korjaukset sanallisesti: 
                    [lista tarvittavista muutoksista tekstiin]
        
                    {selko_identifier}
                    [teksti selkosuomeksi]
                    """.replace('{prompt}', prompt).replace('{selko_identifier}', SELKO_LOCATOR).replace('{instruction}',writer_initial_command)

                    time.sleep(30)
                    output_text = get_llm_response(prompt, 'Muunna seuraava teksti selkosuomeksi:\n\n' + input_text, llm)

                    row = {'output_text': output_text,'input_text':input_text,'reference_text':reference_text,
                           'input_text_file': f['regular'], 'reference_text_file': f['selko'],
                           'model':llm['model'],'id':id}
                    evaluation_results = pd.concat([evaluation_results,pd.DataFrame(row,index=[0])],ignore_index=True)
                    evaluation_results.to_pickle(EVAL_FILE)

    except Exception as error:

        print('An exception occurred: {}'.format(error))

        time.sleep(5)

evaluation_results = evaluation_results.drop_duplicates(subset='id')

EVAL_FILE = 'evaluation_results_part2.pickle'
try:
    eval_scores = pd.read_pickle(EVAL_FILE)
    print('continuing evaluation, %i items exists' % len(eval_scores))
except:
    eval_scores = pd.DataFrame(data={'id':[]})
    print('starting evaluation from scratch')

import text_comparison

def evaluate_result(output_text,reference_text,original_text):
    output_text_raw = remove_tags(output_text)
    reference_text_raw = remove_tags(reference_text)
    original_text_raw = remove_tags(original_text)

    time.sleep(10)
    geval_score = text_comparison.get_geval_score(output_text_raw,reference_text_raw)
    jina_score = text_comparison.get_jina_similarity(output_text_raw,reference_text_raw)
    e5_score = text_comparison.get_e5_similarity(output_text_raw, reference_text_raw)
    time.sleep(10)
    openai_embed_score,gpt4_score = text_comparison.get_openai_scores(output_text_raw,reference_text_raw)
    sari_score = text_comparison.get_sari_score(original_text_raw, output_text_raw,reference_text_raw, lemmatize=False)
    sari_score_lemma = text_comparison.get_sari_score(original_text_raw, output_text_raw, reference_text_raw, lemmatize=True)
    time.sleep(10)
    selkomittari_score = text_comparison.get_selkomittari_score(output_text)
    return {
            'e5_embed_score':e5_score,
            'jina_embed_score':jina_score,
            'SARI_score':sari_score,
            'SARI_score_lemma':sari_score_lemma,
            'openai_custom_score':gpt4_score,
            'openai_embed_score':openai_embed_score,
            'openai_selkomittari_score':selkomittari_score,
            'geval_score':geval_score,
            'length_ratio':len(output_text_raw)/len(reference_text_raw)
            }

for row in evaluation_results.iterrows():
    for loop in range(5):
        try:
            if row[1]['id'] not in eval_scores['id'].to_list():
                print('...analyzing id %s' % row[1]['id'])
                output_text=row[1]['output_text']
                input_text = row[1]['input_text']
                reference_text = row[1]['reference_text']
                evals = evaluate_result(output_text, reference_text, input_text)
                evals['id'] = row[1]['id']
                eval_scores = pd.concat([eval_scores,pd.DataFrame(evals,index=[0])],ignore_index=True)
                eval_scores.to_pickle(EVAL_FILE)
        except Exception as error:

            print('An exception occurred: {}'.format(error))
            time.sleep(30)

print('ALL FINISHED')