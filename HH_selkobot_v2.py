from typing import Annotated
import json
import pandas as pd
import sqlite3
import os
import gradio as gr
import autogen
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv('.env')
# .env file is a text file that contains lines:
# OPENAI_API_KEY="[your key]"   <--- so far only this is needed
# ANTHROPIC_API_KEY="[your key]"
# GOOGLE_API_KEY="[your key]"

# CONVERSATION_TYPE option
# AUTOGEN = using Autogen library with "round_robin" circular group conversation
# MANUAL = using directly OpenAI API to send and receive messages (for comparison purposes)
CONVERSATION_TYPE = 'AUTOGEN' #'MANUAL'

MAXIMUM_ROUNDS = 4 # how many rounds maximum of improvement, terminate early if not already finished. The last version of the text is returned.

gpt4_config_mini = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.1,
    "model": 'gpt-4o-mini',
    "timeout": 60,
}
gpt4_config_full = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0.1,
    "model": 'gpt-4o',
    "timeout": 60
}
OUTPUT_PATH = os.getcwd() + os.sep

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

### bot definitions and prompts

def filewriter(content,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
def filereader(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def bot_constructor(evaluator_prompt,writer_prompt,critic_prompt):

    evaluator_system_prompt = '{prompt}'.replace('{prompt}',evaluator_prompt)

    writer_system_prompt = """Kirjoittaja. Olet selkosuomen huippuasiantuntija. Osaat kirjoittaa ja muokata annettua suomenkielistä tekstiä helpommin selkosuomen muotoon.
    Luet annetun tekstin ja mahdollisesti saamasi palautteen ja korjausehdotukset tarkasti. Uudelleenkirjoitat tekstin saamasi palautteen ja selkosuomen kriteerin pohjalta.
    
    {prompt}
    
    # Tehtävä #
    
    Kirjoita uusi versio annetusta tekstistä selkosuomen periaatteiden ja mahdollisen palautteen perusteella. Saat muokata annettua alkuperäistekstiä, mutta ET KOSKAAN keksiä uutta sisältöä tai faktoja, joita alkuperäisessä tekstissä ei kerrota.
    
    Erota otsikko, ingressi, väliotsikot ja lainaukset aina seuraavilla tageilla, joita ovat:
     otsikko: <headline>...</headline>
     ingressi: <lead>...</lead>
     väliotsikko: <subtitle>...</subtitle>
     lainaukset: <quote>...</quote> 
    Nämä eivät ole osa tekstiä, vaan niitä käytetään tekstin ladontaan.
    
    # Vastaus #
    
    Tarvittavat muutokset sanallisesti: 
    [lista tarvittavista muutoksista tekstiin]
    
    Selkokielinen teksti:
    [teksti selkosuomeksi]
    """.replace('{prompt}',writer_prompt)

    critic_system_prompt = """Kriitikko. Olet selkosuomen huippuasiantuntija ja opettaja. Tunnet perinpohjaisesti selkosuomen kriteerit ja osaat arvioida tekstien selkeyttä, antaa rakentavaa palautetta ja korjausehdotuksia. 
    Tehtävänäsi on analysoida annettua tekstiä ja arvioida täyttääkö se selkosuomen keskeiset kriteerit. Ohjeistat kirjoittajaa tekstin parantamisessa ja puutteiden korjaamisessa.
    
    {prompt}
    
    # Tehtävä #
    
    Arvioi annettua tekstiä selkosuomen periaatteiden pohjalta ja listaa kaikki välttämättömät korjaukset tekstiin. Anna ehdotuksesi aina seuraavassa muodossa. 
    
    # Vastaus #
    
    Arvioni tekstistä:
    [yleinen arviosi tekstistä ja sen selkeydestä]
    
    Korjausehdotukseni:
    [yksityiskohtainen lista korjausehdotuksista tekstiin]
    """.replace('{prompt}',critic_prompt)

    editor_system_prompt="""Editori. Olet uutismedian päätoimittaja ja kirjoitetun viestinnän kokenut huippuasiantuntija. Tehtävänäsi on päättää milloin annetun tekstin hienosäätäminen voidaan lopettaa ja teksti on valmis julkaistavaksi yleisölle.
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

    fact_checker_prompt = '''Faktantarkastaja. Olet faktantarkastaja, jonka tehtävä on tarkastaa, että uusi muunnettu teksti vastaa keskeisten FAKTOJEN osalta alkuperäistä tekstiä. Et ota kantaa tekstin luettavuuteen tai selkeyteen, ainoastaan asiavirheisiin. 
    Käyt yksityskohtaisesti kaikki tekstissä esiintyvät faktat läpi, listaat ne ja varmistat, että faktat eivät oleellisesti muutu tai vääristy uudessa tekstissä. Asiat voidaan kertoa eri tavalla ja eri muodoissa, mutta keskeisten faktojen on aina pysyttävä ennallaan
    Esimerkkejä keskeisistä faktoista:
    -lukumäärät ja numerot
    -erisnimet
    -tittelit ja ammatit
    -tapahtumat
    -väitteet
    -ajankohdat
    
    Kun olet lukenut alkuperäisen ja uuden tekstin huolellisesti läpi, annat palautteesi ja korjausehdotuksesi. Et kirjoita uutta versiota tekstistä, ainoastaan ehdotat kriittisiä korjauksia tekstin uuteen versioon.
    
    # Alkuperäinen teksti (lähdemateriaali) #
    
    "{old_text}"
    
    # Uusi teksti (kohdemateriaali) #
    
    "{new_text}"
    
    # Vastaus # 
    
    arvioni tekstistä:
    [yhteenveto uuden tekstin faktojen paikkansapitävyydestä]
    
    korjausehdotukset:
    [lista tarvittavista faktojen korjauksista uuteen tekstiin]
    '''

    def term_msg(msg):
        return "TERMINATE" in msg["content"]

    def nullify_history(processed_messages):
        processed_messages = [processed_messages[-1]]
        processed_messages[0]['content'] = ''
        return processed_messages

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    )

    fact_checker = autogen.AssistantAgent( # autogen.AssistantAgent(
        name="Faktantarkastaja",
        llm_config=gpt4_config_full,
        description="Faktantarkastaja. Vertailee kahta annettua tekstiä toisiinsa ja tutkii ovatko ne faktojen osalta yhdenmukaiset.",
        is_termination_msg=None,#term_msg,
        system_message=fact_checker_prompt,
    )
    fact_checker.register_hook("process_all_messages_before_reply",nullify_history)

    def write_textfile(input_text: Annotated[str, "input text to write on disk"]):
        filename = OUTPUT_PATH + 'output_file.txt'
        filewriter(input_text,filename)
        print(f"Text successfully written to {filename}")

    latest_text = ''
    def post_message_processor(sender,message,recipient,silent):
        global latest_text
        identifier = '# selkokielinen teksti #'
        if sender.name == 'Kirjoittaja':
            ind = message.find(identifier)
            if ind>-1:
                latest_text = message[(ind + len(identifier)):].strip()
                current_prompt = fact_checker.system_message
                fact_checker.update_system_message(current_prompt.replace('{new_text}',latest_text))
                write_textfile(latest_text)
            else:
                print('\n!!!!!! No new text provided by writer !!!!!!!!\n')
        return message

    writer = autogen.AssistantAgent(# autogen.AssistantAgent(
        name="Kirjoittaja",
        llm_config=gpt4_config_full,
        description="Kirjoittaja ja selkosuomen huippuasiantuntija. Osaa muokata annettua suomenkielistä tekstiä helpommin luettavaan muotoon eli selkosuomentaa.",
        is_termination_msg=None,#term_msg,
        system_message=writer_system_prompt,
    )
    writer.register_hook('process_message_before_send',post_message_processor)

    critic = autogen.AssistantAgent(
        name="Kriitikko",
        llm_config=gpt4_config_full,
        description="Kriitikko. Selkosuomen huippuasiantuntija ja opettaja. Osaa arvioida tekstin selkeyttä ja antaa tarvittaessa rakentavaa palautetta ja korjausehdotuksia tekstiin.",
        system_message=critic_system_prompt,
        is_termination_msg=None, #lambda msg: "TERMINATE" in msg["content"],
    )
    editor = autogen.AssistantAgent(
        name="Editori",
        llm_config=gpt4_config_full,
        description="Editori. Viestinnän asiantuntija ja kokenut päätoimittaja, joka lukee selkosuomeksi kirjoitetun tekstin ja arvioijien kommentit ja korjausehdotukset. Päättää voidaanko teksti julkaista vai pitääkö teksti kirjoittaa uudelleen korjattuna.",
        system_message=editor_system_prompt,
        is_termination_msg=term_msg,
    )

    evaluator = autogen.AssistantAgent(
        name="Evaluaattori",
        llm_config=gpt4_config_full,
        description="Evaluaattori. Selkosuomen huippuasiantuntija, joka pisteyttää annetun tekstin selkomittaria käyttäen.",
        system_message=evaluator_system_prompt,
        is_termination_msg=term_msg,
    )

    return {'evaluator':evaluator,
            'editor':editor,
            'critic':critic,
            'writer':writer,
            'fact_checker':fact_checker}

bots = bot_constructor()
writer= bots['writer']
critic= bots['critic']
fact_checker= bots['fact_checker']
editor= bots['editor']

# Function to process input and display output
def process_text(input_text): #,output_box, console_box):
    input_text = input_text.strip()
    if len(input_text)<50:
        print('too short input text!')
        return

    print(f'Starting processing with CONVERSATION_TYPE = {CONVERSATION_TYPE}')

    current_text = '[text missing]'

    if CONVERSATION_TYPE=='AUTOGEN':

        with open("logs.db", "w") as f:
            f.write("")
        logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
        print("Started Logging session ID: " + str(logging_session_id))

        fact_checker.update_system_message(fact_checker_prompt.replace('{old_text}', input_text))

        agents_list = [user_proxy, writer, critic, fact_checker, editor]
        groupchat = autogen.GroupChat(
            agents=agents_list, messages=[], allow_repeat_speaker=False, max_round=MAXIMUM_ROUNDS*len(agents_list),
            select_speaker_message_template='You are in a role play game. The following roles are available: {roles}. Read the following conversation. Then select the next role from {agentlist} to play. Only return the role.',
            send_introductions=True, speaker_selection_method='round_robin'
        )
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config_mini)
        user_proxy.initiate_chat(
            manager,
            message=f"Käännä seuraava teksti selkosuomeksi:\n\n{input_text}",
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
        final_output = '[not found]'
        for k,row in enumerate(log_data_df.iterrows()):
            console_text += f'\n\n-------------------------- iteration {k+1}----------------------------\n\n'
            role = row[1]['request'].split('.')[0]
            console_text += 'Current speaker: ' + role
            response = '\nResponse: [no response]' if row[1]['response'] is None else '\nResponse:\n\n' + row[1]['response']
            if 'kirjoittaja' in role.lower():
                ind = response.find('# selkokielinen teksti #')
                if ind>-1:
                    final_output = response[(ind+len('# selkokielinen teksti #')):].strip()
            console_text += response

        output_text = final_output

    elif CONVERSATION_TYPE=='MANUAL':

        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        def get_payload(input_str,separator):
            ind = input_str.find(separator)
            assert ind>-1,f'Response not in correct form! Separator not present: {separator}'
            return input_str[(ind+len(separator)):].strip()

        console_text = ''
        current_round = 1
        while current_round<20:
            print(f'Starting round {current_round}')
            console_text += (f'>>>>>>>>>>>>>>>>>> ROUND = {current_round} <<<<<<<<<<<<<<<<<<<<<\n')
            yield None,console_text

            if current_round>1:
                writer_response = client.chat.completions.create(
                    messages=[
                        {"role": "system","content": writer_system_prompt},
                        {
                            "role": "user",
                            "content": ('### ALKUPERÄINEN TEKSTI ###\n\n' + input_text +
                            '\n\n### SELKOKIELINEN TEKSTI ###\n' + current_text +
                            '\n\n### KORJAUSEHDOTUKSET ###\n' + all_suggestions),
                        }
                    ],
                    model=gpt4_config_full['model'],
                    temperature=gpt4_config_full['temperature'],
                )
            else:
                writer_response = client.chat.completions.create(
                    messages=[
                        {"role": "system","content": writer_system_prompt},
                        {
                            "role": "user",
                            "content": f"Käännä seuraava teksti selkosuomeksi:\n\n{input_text}",
                        }
                    ],
                    model=gpt4_config_full['model'],
                    temperature=gpt4_config_full['temperature'],
                )
            current_text = get_payload(writer_response.choices[0].message.content,"# selkokielinen teksti #")
            print(f'...writer responded')

            console_text += ('\n\n------------------------------------------------------- speaker = WRITER -------------------------------------------------------\n\n' + writer_response.choices[0].message.content.strip())
            yield current_text,console_text

            critic_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": critic_system_prompt},
                    {
                        "role": "user",
                        "content": current_text,
                    }
                ],
                model=gpt4_config_full['model'],
                temperature=gpt4_config_full['temperature'],
            )
            suggestions1 = get_payload(critic_response.choices[0].message.content, "# korjausehdotukset #")
            print(f'...critic responded')

            console_text += ('\n\n------------------------------------------------------- speaker = CRITIC -------------------------------------------------------\n\n' + critic_response.choices[0].message.content.strip())
            yield current_text,console_text

            factcheck_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": fact_checker_prompt},
                    {
                        "role": "user",
                        "content": '### ALKUPERÄINEN TEKSTI ###\n' + input_text + '\n\n### SELKOKIELINEN TEKSTI ###\n' + current_text,
                    }
                ],
                model=gpt4_config_full['model'],
                temperature=gpt4_config_full['temperature'],
            )
            suggestions2 = get_payload(factcheck_response.choices[0].message.content, "# korjausehdotukset #")
            print(f'...fact-checker responded')

            console_text += ('\n\n------------------------------------------------------- speaker = FACT-CHECKER -------------------------------------------------------\n\n' + factcheck_response.choices[0].message.content.strip())
            yield current_text,console_text

            all_suggestions = '### Kriitikon korjausehdotukset ###\n' + suggestions1 + '\n\n### Faktantarkastajan korjausehdotukset ###\n' + suggestions2

            editor_response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": editor_system_prompt},
                    {
                        "role": "user",
                        "content": '### SELKOKIELINEN TEKSTI ###\n\n' + current_text + '\n\n### KORJAUSEHDOTUKSET ###\n' + all_suggestions
                    }
                ],
                model=gpt4_config_full['model'],
                temperature=gpt4_config_full['temperature'],
            )
            print(f'...editor responded')
            console_text += ('\n\n------------------------------------------------------- speaker = EDITOR -------------------------------------------------------\n\n' + editor_response.choices[0].message.content.strip() + '\n\n')
            yield current_text,console_text

            if 'TERMINATE' in editor_response.choices[0].message.content:
                print(f'...decision=TERMINATE')
                break
            else:
                print(f'...decision=rewrite')

            if current_round==MAXIMUM_ROUNDS:
                print(f'\nMAXIMUM ROUNDS REACHED, FORCE TERMINATION!\n')
                break

            current_round+=1

            output_text = current_text

    else:
        raise Exception('UNKNOWN AGENT TYPE')

    print('Processing finished')

    yield output_text,console_text

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
