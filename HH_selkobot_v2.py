from typing import Annotated
import json
import pandas as pd
import sqlite3
import os
import gradio as gr
import autogen
from autogen import AssistantAgent, UserProxyAgent
from openai import OpenAI
import numpy as np
from anthropic import Anthropic
import asyncio
from typing import List, Sequence
from dotenv import load_dotenv
import itertools
load_dotenv('.env')
# .env file is a text file that contains lines:
# OPENAI_API_KEY="[your key]"   <--- so far only this is needed
# ANTHROPIC_API_KEY="[your key]"
# GOOGLE_API_KEY="[your key]"

# CONVERSATION_TYPE option
# AUTOGEN = using Autogen library with "round_robin" circular group conversation
# MANUAL = using directly OpenAI API to send and receive messages (for comparison purposes)

MAXIMUM_ROUNDS = 4

gpt4_config_mini = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.05,
    "model": 'gpt-4o-mini',
    "timeout": 60,
    #"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"
}
gpt4_config_full = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.05,
    "model": 'gpt-4o',
    "timeout": 60,
    "base_url":"http://localhost:1234/v1","api_key":"lm-studio"
}
claude_config = {
    "temperature": 0.05,
    "model": "claude-3-5-sonnet-20241022",
    "timeout": 60,
    "api_type": "anthropic",
    "api_key": os.getenv('ANTHROPIC_API_KEY')
}

#client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

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
 ...

 Faktoiksi EI lasketa kielellisiä ilmaisuja, tekstin muotoilua tai lausejärjestystä, jotka eivät vaikuta faktoihin.

 Älä huomioi teksteissä olevia tageja:
 <headline>...</headline>
 <lead>...</lead>
 <subtitle>...</subtitle>
 <quote>...</quote> 
 Ne eivät ole osa tarkastettavaa tekstiä, vaan niitä käytetään ainoastaan tekstin ladontaan.

 # Alkuperäinen teksti (lähdemateriaali) #

 {old_text}

 # Tehtävä #

 Vedä syvään henkeä ja lue alkuperäinen teksti (lähdemateriaali) ja annettu uusi teksti (kohdemateriaali) huolellisesti läpi. 
 Vertaile alkuperäisessä ja uudessa tekstissä kerrottuja faktoja toisiinsa. Älä keskity kielelliseen ulkoasuun, ainoastaan koviin faktoihin.
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

### bot definitions and prompts

SELKO_LOCATOR = 'Teksti selkosuomeksi:'

def filewriter(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def filereader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()
def process_all_messages_before_reply_hook(processed_messages):
    global latest_test
    if processed_messages[-1]['name'] == 'Kirjoittaja':
        processed_messages = [processed_messages[-1]]  # take only the latest message as a reference
        if len(latest_text) < 10:
            raise (Exception('Latest text is empty!'))
        processed_messages[0]['content'] = 'Tarkasta tämä teksti (kohdemateriaali):\n\n' + latest_text  # nullify content
    return processed_messages

def post_message_processor(sender, message, recipient, silent):
    global console_text
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
    console_text += '\n\n' + message
    return message


def get_selkomittari_score(model_prediction):
    with open(os.getcwd() + r'\prompts\selkomittari_evaluator_JSON.txt','r',encoding='utf-8') as f:
        evaluator_system_prompt = f.read()

    openai_client = OpenAI()

    scores = []
    for iteration in range(2):
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": evaluator_system_prompt},
                {
                    "role": "user",
                    "content": 'Arvio seuraava teksti:\n\n' + model_prediction,
                }
            ],
            **gpt4_config_full
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
            {'items':[4,5,6]},
            {'items': [7]},
            {'items': list(range(8,13))},
            {'items': [13,14]},
            {'items': list(range(20,25))},
            {'items': list(range(25,29))},
            {'items': list(range(34,40))},
            {'items': list(range(40,48))},
            {'items': list(range(48,53))}
        ]
        temp = list(itertools.chain.from_iterable([x['items'] for x in subsection_scores]))
        main_scores = [x for x in range(1,53) if x not in temp]
        for k in range(0,len(subsection_scores)):
            subsection_scores[k]['scores']=[]

        main_score = 0
        main_count = 0
        apu_score = 0
        apu_count = 0
        scoremap = {'1':3.2,'0':0.8,1:3.2,0:0.8}
        for x in scores:
            num = int(x['kriteeri'])
            if x['pisteet'] > 0:
                if num in main_scores:
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

        score = (main_score + apu_score) / (main_count + apu_count)
        scores.append(score)

    score = np.mean(scores)
    return score

def bot_constructor(writer_prompt=None, critic_prompt=None, factchecker_prompt=None, editor_prompt=None, llm=None):
    factchecker_prt = factchecker_prompt
    editor_prt = editor_prompt

    writer_prt = """Kirjoittaja. Olet selkosuomen huippuasiantuntija. Osaat kirjoittaa ja mukauttaa annettua suomenkielistä tekstiä helpompaan selkosuomen muotoon.

    {prompt}

    # TEHTÄVÄ #

    Kirjoita selkosuomeksi mukautettu versio annetusta yleiskielisestä uutistekstistä. 
    {instruction} 
    Saat muokata annettua yleiskielistä tekstiä, mutta ET SAA keksiä uutta sisältöä tai uusia faktoja, joita alkuperäisessä tekstissä ei kerrota.

    Erota tekstissäsi otsikko, ingressi, väliotsikot ja lainaukset aina seuraavilla tageilla, joita ovat:
     otsikko: <title>...</title>
     ingressi: <lead>...</lead>
     väliotsikko: <subtitle>...</subtitle>
     lainaukset: <quote>...</quote> 
    JOKAINEN teksti sisältää otsikon ja ingressin. Nämä eivät ole osa tekstiä, vaan niitä käytetään tekstin ladontaan verkkosivulle.

    # VASTAUS #

    Anna vastauksesi täsmälleen seuraavassa muodossa

    Tarvittavat muutokset/korjaukset sanallisesti: 
    [yksityiskohtainen lista tarvittavista muutoksista tekstiin]

    {selko_identifier}
    [teksti selkosuomeksi]
    """.replace('{prompt}', writer_prompt).replace('{selko_identifier}', SELKO_LOCATOR).replace('{instruction}',
                                                                                                writer_initial_command)

    critic_prt = """Kriitikko. Olet selkosuomen huippuasiantuntija ja opettaja. Tunnet perinpohjaisesti selkosuomen kriteerit ja osaat arvioida tekstien selkeyttä, antaa rakentavaa palautetta ja korjausehdotuksia. 
    Tehtävänäsi on analysoida annettua tekstiä ja arvioida täyttääkö se selkosuomen keskeiset kriteerit. Ohjeistat kirjoittajaa tekstin parantamisessa ja puutteiden korjaamisessa.

    {prompt}

    Jokainen teksti sisältää aina otsikon ja ingressin eli tiivistelmäkappaleen. Lisäksi tekstissä voi olla väliotsikoita ja lainauksia. Kaikki nämä neljä kenttää on merkitty tageilla, joita ovat:
        otsikko: <title>...</title>
        ingressi: <lead>...</lead>
        väliotsikko: <subtitle>...</subtitle>
        lainaukset: <quote>...</quote>
    Nämä tagit eivät koskaan näy lukijalle tai ole osa tekstiä, vaan ne on lisätty selkeyden ja ladonnan vuoksi.

    # Tehtävä #

    Arvioi annettua tekstiä selkosuomen periaatteiden pohjalta ja listaa kaikki välttämättömät korjaukset tekstiin. Anna ehdotuksesi aina seuraavassa muodossa. 

    # Vastaus #

    Anna vastauksesi täsmälleen seuraavassa muodossa

    Arvioni tekstistä:
    [yleinen arviosi tekstistä ja sen selkeydestä]

    Korjausehdotukseni:
    [yksityiskohtainen lista korjausehdotuksista tekstiin]
    """.replace('{prompt}', critic_prompt)

    def term_msg(msg):
        return "TERMINATE" in msg["content"]

    fact_checker = AssistantAgent(  # autogen.AssistantAgent(
        name="Faktantarkastaja",
        llm_config=llm,
        description="Faktantarkastaja. Vertailee kahta annettua tekstiä toisiinsa ja tutkii ovatko ne faktojen osalta yhdenmukaiset.",
        is_termination_msg=None,  # term_msg,
        system_message=factchecker_prt,
    )
    fact_checker.register_hook("process_all_messages_before_reply", process_all_messages_before_reply_hook)
    fact_checker.register_hook('process_message_before_send', post_message_processor)

    writer = AssistantAgent(  # autogen.AssistantAgent(
        name="Kirjoittaja",
        llm_config=llm,
        description="Kirjoittaja ja selkosuomen huippuasiantuntija ja toimittaja. Osaa muokata annettua suomenkielistä tekstiä helpommin luettavaan muotoon eli selkosuomentaa.",
        is_termination_msg=None,  # term_msg,
        system_message=writer_prt,
    )
    writer.register_hook('process_message_before_send', post_message_processor)
    writer.register_hook("process_all_messages_before_reply", process_all_messages_before_reply_hook)

    critic = AssistantAgent(
        name="Kriitikko",
        llm_config=llm,
        description="Kriitikko. Selkosuomen huippuasiantuntija ja opettaja. Osaa arvioida tekstin selkeyttä ja antaa tarvittaessa rakentavaa palautetta ja korjausehdotuksia tekstiin.",
        system_message=critic_prt,
        is_termination_msg=None,  # lambda msg: "TERMINATE" in msg["content"],
    )
    critic.register_hook('process_message_before_send', post_message_processor)
    critic.register_hook("process_all_messages_before_reply", process_all_messages_before_reply_hook)

    editor = AssistantAgent(
        name="Editori",
        llm_config=llm,
        description="Editori. Uutismedian päätoimittaja ja kirjoitetun viestinnän kokenut huippuasiantuntija, joka lukee selkosuomeksi kirjoitetun tekstin ja arvioijien kommentit ja korjausehdotukset. Päättää voidaanko teksti julkaista vai pitääkö teksti kirjoittaa uudelleen korjattuna.",
        system_message=editor_prt,
        is_termination_msg=term_msg,
    )
    editor.register_hook('process_message_before_send', post_message_processor)
    editor.register_hook("process_all_messages_before_reply", process_all_messages_before_reply_hook)

    evaluator = autogen.AssistantAgent(
       name="Evaluaattori",
       llm_config=llm,
       description="Evaluaattori. Selkosuomen ja viestinnän huippuasiantuntija, joka pisteyttää annetun tekstin selkomittaria käyttäen.",
       system_message=evaluator_prt,
       is_termination_msg=None,
    )
    evaluator.register_hook('process_message_before_send', post_message_processor)
    evaluator.register_hook("process_all_messages_before_reply", process_all_messages_before_reply_hook)

    return {  # "'evaluator':evaluator,
        'editor': editor,
        'critic': critic,
        'writer': writer,
        'evaluator': evaluator,
        'fact_checker': fact_checker}

writer_system_prompt = filereader(os.getcwd() + r'/prompts/HH_selkobot_v2.txt')
critic_system_prompt = writer_system_prompt

bots = bot_constructor(writer_prompt=writer_system_prompt, critic_prompt=critic_system_prompt,
                       factchecker_prompt=factchecker_system_prompt, editor_prompt=editor_system_prompt,
                       llm=gpt4_config_mini)
writer = bots['writer']
critic = bots['critic']
factchecker = bots['fact_checker']
editor = bots['editor']
evaluator = bots['evaluator']
user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=None,
    code_execution_config=False,
)
agents_list = [user_proxy,writer,factchecker,editor]

# Function to process input and display output
def process_text(input_text): #,output_box, console_box):
    global latest_text
    global console_text
    global writing_loop

    input_text = input_text.strip()
    if len(input_text)<50:
        print('too short input text!')
        return

    print(f'Starting processing')

    latest_text = 'EMPTY'
    console_text = ''
    writing_loop = 0

    factchecker_system_prompt0 = factchecker_system_prompt.replace('{old_text}', input_text)
    factchecker.update_system_message(factchecker_system_prompt0)

    groupchat = autogen.GroupChat(
        agents=agents_list, messages=[], allow_repeat_speaker=False, max_round=MAXIMUM_ROUNDS * len(agents_list),
        select_speaker_message_template='You are in a role play game. The following roles are available: {roles}. Read the following conversation. Then select the next role from {agentlist} to play. Only return the role.',
        send_introductions=False, speaker_selection_method='round_robin'
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config_mini)
    user_proxy.initiate_chat(
        manager,
        message=f"Käännä seuraava teksti selkosuomeksi:\n\n{input_text}",
    )

    print('Processing finished')

    yield latest_text,console_text

# Function to clear the text boxes
def clear_text():
    return "", "", ""  # Return empty strings for all outputs

#
custom_css = """
    /* Custom font size for input and output boxes */
    #INPUTBOX textarea, #OUTPUTBOX textarea {
        font-size: 16px !important;  /* Increase font size for text areas */
        overflow-y: scroll !important;  /* Ensure vertical scrollbar is always visible */
        scrollbar-width: thin;  /* Thin scrollbar for Firefox */
    }

    /* Center title and adjust font size */
    #app-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Ensure the console has a scrollbar and custom font size */
    #CONSOLE textarea {
        font-size: 16px !important;
        overflow-y: scroll !important;  /* Enable vertical scrollbar */
        scrollbar-width: thin;  /* Thin scrollbar for Firefox */
    }

    /* Custom scrollbar styles for WebKit browsers */
    #INPUTBOX::-webkit-scrollbar, 
    #OUTPUTBOX::-webkit-scrollbar, 
    #CONSOLE::-webkit-scrollbar {
        width: 8px;  /* Width of the scrollbar */
    }

    #INPUTBOX::-webkit-scrollbar-thumb, 
    #OUTPUTBOX::-webkit-scrollbar-thumb, 
    #CONSOLE::-webkit-scrollbar-thumb {
        background-color: #b0b0b0;  /* Color of the scrollbar thumb */
        border-radius: 10px;
    }

    #INPUTBOX::-webkit-scrollbar-track, 
    #OUTPUTBOX::-webkit-scrollbar-track, 
    #CONSOLE::-webkit-scrollbar-track {
        background-color: #f0f0f0;  /* Color of the scrollbar track */
    }
"""

# Creating the Gradio interface
with gr.Blocks(css=custom_css, fill_width=True) as demo:
    gr.HTML("<div id='app-title'>HH SelkoBot</div>")

    with gr.Row():
        input_box = gr.Textbox(
            label="INPUT",
            placeholder='Enter input text here',
            value=DEFAULT_TEXT,
            lines=35,
            elem_id="INPUTBOX",
            show_label=True,
            interactive=True,
            scale=5,
            min_width=300
        )
        output_box = gr.Textbox(
            label="OUTPUT",
            placeholder="",
            lines=35,
            elem_id="OUTPUTBOX",
            show_label=True,
            interactive=False,
            scale=5,
            min_width=300
        )

    with gr.Row():
        process_btn = gr.Button("Process", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    with gr.Row():
        console_box = gr.Textbox(
            label="Console",
            placeholder="Console output will be displayed here...",
            lines=30,
            interactive=False,
            elem_id="CONSOLE",
            scale=10,  # Full width under buttons
            min_width=300  # Ensures a minimum width for readability
        )

    # Define the button click events
    #process_btn.click(lambda x: process_text(x, output_box, console_box), inputs=input_box, outputs=[console_box])
    process_btn.click(fn=process_text, inputs=input_box, outputs=[output_box, console_box])
    clear_btn.click(fn=clear_text, outputs=[input_box, output_box, console_box])

# Launch the interface
demo.launch()
