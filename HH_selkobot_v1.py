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

gpt4_config_mini = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "model": 'gpt-4o-mini',
    "timeout": 60,
}
gpt4_config_full = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "model": 'gpt-4o',
    "timeout": 60
}

OUTPUT_PATH = os.getcwd() + os.sep

# prefill with this news text for fast testing purposes
DEFAULT_TEXT = '''Puutarhajätettä kärrätty luvatta luontoon Vantaalla – "Tuntuu olevan hyvä omenavuosi, joten ilmoituksia voi tulla lisääkin"

Vantaan ympäristökeskus muistuttaa, että puutarhajäte on tärkeää kierrättää oikein, jotta vieraslajit eivät leviä.

Vantaan ympäristökeskus on saanut ilmoituksia puutarhajätteestä, joka on kipattu sille kuulumattomille paikoille. Vantaan ympäristökeskuksen johtavan ympäristötarkastaja Maarit Rantataron mukaan erilaista puutarhajätettä on kipattu metsän laitaan tai kadun varteen. Puutarhajätteen kippaamisesta luontoon ja yleisille alueille tulee Rantataron mukaan vuosittain 10–15 valitusta. Tänä vuonna ilmoituksia on tullut elokuun alkuun mennessä lähemmäs 10. – Näyttää siltä, että tänä vuonna ilmoituksia tulee enemmän kuin aiemmin.

Rantataro sanoo, että ilmoitukset liittyvät useimmiten siihen, että lähimetsän laitaan tai ojaan on kuljetettu erilaista puutarhajätettä, kuten risuja, lehtiä, kukkaruukkuja ja joskus jopa joulukuusia.

Kyse on jäterikkomuksesta, josta voi tulla sakkoa.

Viime viikolla ympäristökeskukseen tuli ensimmäinen ilmoitus omenoista, jotka oli kärrätty omalta kotipihalta pois. – Tuntuu olevan hyvä omenavuosi, joten ilmoituksia voi tulla lisääkin, Rantataro arvelee.

Puusta pudonneet, käyttämättömät omenat ovat puutarhajätettä, jotka tulee kompostoida tai viedä sortti-asemalle. – Omenat on syytä kompostoida lämpöeristetyssä kompostorissa. Omenia voi laittaa myös tarjolle oman pihakadun varteen.

Rantataro tietää myös tapauksia, joissa puutarhajäte on päätetty kaataa naapurin puolelle. Hän muistuttaa, että puutarhajäte on tärkeää kierrättää oikein, jotta vieraslajit eivät leviä. Omenat ja muut hedelmät voivat puolestaan kerätä rottia. – Jos joku kaataa puutarhajätteen paikkaan, johon se ei kuulu, on hän jätelain nojalla velvollinen sen korjaamaan. Kyse on jäterikkomuksesta, josta voi tulla sakkoa.

Vantaan viranomaiset kiertävät pian talojen pihoilla: Levällään lojuvat lelut eivät tuo huomautusta, mutta moni muu asia tuo

Viranomaiset antoivat törkyisistä pihoista runsaasti huomautuksia Vantaalla – "Pressu ei pelasta tilannetta, jos roinaa on kertynyt paljon"

Päiväkummun proffan piha on laboratorio – "Tavoitteeni on kiertopuutarha"'''

### bot definitions and prompts

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    code_execution_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)

writer_system_prompt = """Kirjoittaja. Olet selkosuomen asiantuntija eli osaat muokata annettua suomenkielistä tekstiä helpommin luettavaan muotoon eli SELKOSUOMENTAA.
Luet annetun tekstin ja mahdollisesti saamasi palautteen ja korjausehdotukset tarkasti. Uudelleenkirjoitat tekstin saamasi palautteen ja seuraavien 55 kriteerin pohjalta:

1. Teksti on kokonaisuutena arvioituna yleiskieltä helpompaa.
2. Aihetta käsitellään lukijan kannalta mielekkäästä näkökulmasta.
3. Aihetta käsitellään konkreettisella, havainnollisella tavalla käyttämällä muun muassa arkielämää lähellä olevia esimerkkejä.
4. Tekstissä ei ole sisällöllisiä aukkoja. Lukija saa joka kohdassa tekstin ymmärtämisen kannalta riittävästi tietoa.
5. Teksti selittää itse itsensä eikä nojaa liikaa lukijan yleistietoon tai muiden tekstien tuntemukseen.
6. Tekstistä käy selvästi ilmi sen viestinnällinen tavoite: pyrkiikö teksti esimerkiksi vaikuttamaan lukijaan, välittämään tietoa tai ohjeistamaan.
7. Aihetta käsitellään pääosin konkreettisten toimijoiden ja ihmisten kautta (hakija, poliisi, me). Toimijoina on vain vähän abstrakteja substantiiveja (suunnitelma koskee, asiakaslähtöisyys on, avoimuus toteutuu).
8. Tekstissä ei ole epäolennaista tietoa.
9. Teksti ei ole liian tiivistä; myöskään yhteen lauseeseen ei ole pakattu liikaa asiaa.
10. Tekstin sävy on tilanteeseen sopiva.
11. Teksti ei aliarvioi lukijaa. Se ei esimerkiksi selitä liikaa tai ole liian opettavainen. Se on myös kieleltään kohderyhmän ikätasolle sopiva.
12. Teksti on suunnattu selkeästi lukijalle esimerkiksi suoran puhuttelun avulla (esim. pronomini sinä, yksikön toinen persoona kirjoita tai omistusliite nimesi).
13. Lukijalle tarkoitetut toimintaohjeet ilmaistaan selkeästi ja yksiselitteisesti. Tekstissä erotetaan kielellisesti, mikä on lukijalle pakollista (täytyä, pitää), mikä taas mahdollista tai suositeltavaa (voida, kannattaa). Lukijalle suunnattuja ohjeita ei esitetä passiivimuodossa (lomake täytetään).
14. Asioista kerrotaan yleisellä tasolla tai tekstissä hyödynnetään epäsuoraa puhuttelua silloin, kun suora puhuttelu ei tunnu luontevalta esimerkiksi tekstin tyylin, tekstilajin, tekstissä käsiteltävien arkaluonteisten puheenaiheiden tai liiallisen suoran puhuttelun takia. (Huumetestiin voi ilmoittautua verkossa.)
15. Lukijaa ei esitetä liian usein passiivisena tai avun kohteena, vaan lukija on myös aktiivinen toimija.
16. Tekstissä on pääosin yleistä, lukijoille tutuksi arvioitua sanastoa.
17. Jos vaikeaa sanaa ei voi korvata tai välttää, se on selitetty lukijalle ymmärrettävällä ja kontekstiin sopivalla tavalla.
18. Tekstissä ei ole huomattavan paljon pitkiä sanoja.
19. Tekstissä on pääosin konkreettisia sanoja (kirjoittaa, täytyä, sairaala).
20. Toistoa käytetään siten, että se lisää tekstin ymmärrettävyyttä. Samaan asiaan ei viitata liian monella eri tavalla tai synonyymillä.
21. Selitys on kohdassa, jossa vieraaksi oletettu sana esiintyy ensimmäistä kertaa. Selitys ei aiheuta lisäselittämisen tarvetta. Pitkässä tekstissä selitys toistetaan tarvittaessa.
22. Tekstissä ei selitetä sanoja, jotka voidaan olettaa lukijalle tutuksi (sairaala on paikka, jossa hoidetaan potilaita).
23. Tekstissä on käytetty vain olennaisia erikoiskielisiä käsitteitä, ja ne on selitetty hyvin.
24. Pitkissä yhdyssanoissa on hyödynnetty osittaistoistoa ensimaininnan jälkeen (maa- ja metsätalousministeriö > ministeriö).
25. Pronominiviittaukset ovat selviä, eikä pronomini jää liian kauaksi viittauskohteestaan.
26. Tekstissä ei ole vierassanoja, jos niille on yleinen, kotoperäinen vastine (resurssi – aikaa, rahaa, työtä, ihmisiä; informaatio – tieto; reklamoida – tehdä valitus; show – esitys).
27. Kielikuvia on maltillisesti. Tekstissä käytetyt kielikuvat ovat tuttuja ja yleisiä, ja niitä on vaikea korvata muilla sanoilla (säästää aikaa, sähkövirta, verkko).
28. Tekstissä ei ole kielikuvia, joiden ymmärtäminen vaatii luovaa päättelyä (juustohöylätä, aivovuoto, lasikatto, silmiinpistävä, tiekartta merkityksessä ’suunnitelma’).
29. Tekstissä on isoja ja tarkkoja lukuja vain, jos se on tekstin aiheen kannalta perusteltua. Lukuja on tarvittaessa likimääräistetty.
30. Luvut, lukumäärät, mittayksiköt ja lukujen väliset suhteet esitetään havainnollisesti.
31. Tekstissä ei ole lyhenteitä, pois lukien vakiintuneet lyhenteet, jotka tunnistetaan paremmin lyhenteinä kuin aukikirjoitettuina (Kela, PDF, DVD).
32. Tekstissä ei ole huomattavan paljon vaikeaksi arvioitavia kielen rakenteita.
33. Lauseet ja virkkeet ovat pääosin lyhyitä.
34. Yhdessä lauseessa ilmaistaan vain yksi tärkeä asia.
35. Substantiiveilla ei ole monimutkaisia määritteitä, kuten partisiippirakenteita (maasta lähteneet henkilöt, koulutukseen liittyvät materiaalit, lääkäriltä saamasi ohje).
36. Tekstissä ei ole lauseenvastikkeita eikä muita vastaavia infiniittisiä rakenteita (Voidakseen osallistua opiskelijan täytyy ilmoittautua etukäteen. Jos haluat terveyttäsi selvitettävän, ota yhteyttä. Allekirjoittamatta jäänyttä hakemusta ei käsitellä.).
37. Tekstissä käytetään nominien perusmuotoja, jos se on lauseyhteydessä mahdollista ja luontevaa.
38. Tekstissä on pääosin nominien helpoimpia taivutusmuotoja. Harvinaisia sijamuotoja abessiivia (huomiotta, tauotta), komitatiivia (liitteineen) ja instruktiivia (pienin muutoksin) ei ole.
39. Tekstissä ei ole sanoja, joissa on useita erilaisia elementtejä, kuten johtimia, taivutuspäätteitä ja liitteitä (lomakkeisiimmekaan, ymmärtääkseni, puolustamiesi).
40. Verbit ovat pääosin preesensissä (lähetät) ja imperfektissä (lähetit). Perfektiä ja pluskvamperfektiä (olet lähettänyt, olit lähettänyt) käytetään vain, jos tekstin aikasuhteet sitä vaativat.
41. Verbin moduksista käytetään enimmäkseen indikatiivia (palautan, puhumme) ja imperatiivin 2. persoonaa (palauta, puhukaa). Tekstissä ei ole harvinaisia verbimoduksia, kuten potentiaalia (tehnee, tietänemme) ja vanhahtavia 3. persoonan imperatiivimuotoja (tehköön).
42. Konditionaalia käytetään vain, jos sitä ei voi korvata indikatiivilla ilman, että merkitys selvästi muuttuu. (Hakemus kannattaa jättää, vaikka et olisi saanut vielä opiskelupaikkaa.)
43. Substantiiveilla ei ole useita määritteitä (Sairastuminen voi olla vakava, vaarallinen ja pelottava asia.).
44. Yhteen kuuluvat sanat, kuten verbiliitot ja verbien rektion mukaiset ilmaukset, esitetään peräkkäin tai mahdollisimman lähekkäin (Päätös vaikuttaa ensi kuun alusta alkaen asumistukeesi. > Päätös vaikuttaa asumistukeesi ensi kuun alusta alkaen.).
45. Tekstissä on pääosin lauseita, jotka rakentuvat persoonamuotoisen verbin varaan (Palauta lomake ajoissa. Vastaamme viesteihin maanantaisin.).
46. Tekstissä ei ole substantiivityylisiä ilmauksia (ns. substantiivitauti, esim. Projektin toteutuksen suunnittelu aikataulutetaan.).
47. Tekstissä käytetään suoraa sanajärjestystä (esim. subjekti, predikaatti, objekti). Käänteistä sanajärjestystä käytetään vain silloin, jos tekstin rakenne niin vaatii tai teksti muuten muuttuu monotoniseksi.
48. Predikaatti sijaitsee lauseen alkupuolella.
49. Tekstissä käytetään passiivia vain silloin, kun tekijä ei ole tiedossa tai tekijän mainitseminen ei ole olennaista (Presidentinvaalit järjestetään kuuden vuoden välein. Talo on rakennettu 1920-luvulla.).
50. Lauseessa ei esitetä kaksinkertaista kieltoa (Laskuja ei saa jättää maksamatta.).
51. Virkerakenteet ovat yksinkertaisia. Sivulauseita on pääosin vain yksi.
52. Virkkeissä päälause ja sivulause ovat tekstin etenemisen kannalta loogisessa järjestyksessä.
53. Virkkeissä lauseet on sidottu toisiinsa esimerkiksi konjunktioilla (siksi, mutta, kun) niin, että asioiden väliset suhteet käyvät ilmi tekstistä.
54. Tekstissä ei ole kiilalauseita (Palveluja, joita ovat esimerkiksi asumispalvelut, vammaispalvelut ja vanhuspalvelut, voit hakea tällä lomakkeella.).
55. Virkkeissä ei esitetä monimutkaisia kieltosuhteita (Et saa kurssimerkintää, jos et palauta tehtävää).

Kirjoita uusi versio annetusta tekstistä näiden kriteerien ja palautteen perusteella. Saat muokata annettua tekstiä, mutta et keksiä uutta sisältöä tai faktoja. 
Anna vastauksesi aina muodossa:

# tarvittavat muutokset #
{lista tarvittavista muutoksista tekstiin}

# selkokielinen teksti #
{uusi teksti}
"""

critic_system_prompt = """Kriitikko. Olet selkosuomen asiantuntija ja kriitikko, joka osaa arvioida tekstin selkeyttä, tuottaa rakentavaa palautetta ja antaa korjausehdotuksia tekstiin. 
Tehtävänäsi on analysoida annettua tekstiä ja arvioida täyttääkö se selkokielen kriteerit. Ohjeistat kirjoittajaa tekstin parantamisessa.
Käytät kaikkia seuraavia kriteereitä tekstin arvioinnissa (esimerkit kerrottu sulkeissa):

1. Tekstissä on pääosin yleistä, lukijoille tutuksi arvioitua sanastoa.
2. Jos vaikeaa sanaa ei voi korvata tai välttää, se on selitetty lukijalle ymmärrettävällä ja kontekstiin sopivalla tavalla.
3. Tekstissä ei ole huomattavan paljon pitkiä sanoja.
4. Tekstissä on pääosin konkreettisia sanoja (kirjoittaa, täytyä, sairaala).
5. Toistoa käytetään siten, että se lisää tekstin ymmärrettävyyttä. Samaan asiaan ei viitata liian monella eri tavalla tai synonyymillä.
6. Selitys on kohdassa, jossa vieraaksi oletettu sana esiintyy ensimmäistä kertaa. Selitys ei aiheuta lisäselittämisen tarvetta. Pitkässä tekstissä selitys toistetaan tarvittaessa.
7. Tekstissä ei selitetä sanoja, jotka voidaan olettaa lukijalle tutuksi (sairaala on paikka, jossa hoidetaan potilaita).
8. Tekstissä on käytetty vain olennaisia erikoiskielisiä käsitteitä, ja ne on selitetty hyvin.
9. Pitkissä yhdyssanoissa on hyödynnetty osittaistoistoa ensimaininnan jälkeen (maa- ja metsätalousministeriö > ministeriö).
10. Pronominiviittaukset ovat selviä, eikä pronomini jää liian kauaksi viittauskohteestaan.
11. Tekstissä ei ole vierassanoja, jos niille on yleinen, kotoperäinen vastine (resurssi – aikaa, rahaa, työtä, ihmisiä; informaatio – tieto; reklamoida – tehdä valitus; show – esitys).
12. Kielikuvia on maltillisesti. Tekstissä käytetyt kielikuvat ovat tuttuja ja yleisiä, ja niitä on vaikea korvata muilla sanoilla (säästää aikaa, sähkövirta, verkko).
13. Tekstissä ei ole kielikuvia, joiden ymmärtäminen vaatii luovaa päättelyä (juustohöylätä, aivovuoto, lasikatto, silmiinpistävä, tiekartta merkityksessä 'suunnitelma').
14. Tekstissä on isoja ja tarkkoja lukuja vain, jos se on tekstin aiheen kannalta perusteltua. Lukuja on tarvittaessa likimääräistetty.
15. Luvut, lukumäärät, mittayksiköt ja lukujen väliset suhteet esitetään havainnollisesti.
16. Tekstissä ei ole lyhenteitä, pois lukien vakiintuneet lyhenteet, jotka tunnistetaan paremmin lyhenteinä kuin aukikirjoitettuina (Kela, PDF, DVD).
17. Tekstissä ei ole huomattavan paljon vaikeaksi arvioitavia kielen rakenteita.
18. Lauseet ja virkkeet ovat pääosin lyhyitä.
19. Yhdessä lauseessa ilmaistaan vain yksi tärkeä asia.
20. Substantiiveilla ei ole monimutkaisia määritteitä, kuten partisiippirakenteita (maasta lähteneet henkilöt, koulutukseen liittyvät materiaalit, lääkäriltä saamasi ohje).
21. Tekstissä ei ole lauseenvastikkeita eikä muita vastaavia infiniittisiä rakenteita (Voidakseen osallistua opiskelijan täytyy ilmoittautua etukäteen. Jos haluat terveyttäsi selvitettävän, ota yhteyttä. Allekirjoittamatta jäänyttä hakemusta ei käsitellä.).
22. Tekstissä käytetään nominien perusmuotoja, jos se on lauseyhteydessä mahdollista ja luontevaa.
23. Tekstissä on pääosin nominien helpoimpia taivutusmuotoja. Harvinaisia sijamuotoja abessiivia (huomiotta, tauotta), komitatiivia (liitteineen) ja instruktiivia (pienin muutoksin) ei ole.
24. Tekstissä ei ole sanoja, joissa on useita erilaisia elementtejä, kuten johtimia, taivutuspäätteitä ja liitteitä (lomakkeisiimmekaan, ymmärtääkseni, puolustamiesi).
25. Verbit ovat pääosin preesensissä (lähetät) ja imperfektissä (lähetit). Perfektiä ja pluskvamperfektiä (olet lähettänyt, olit lähettänyt) käytetään vain, jos tekstin aikasuhteet sitä vaativat.
26. Verbin moduksista käytetään enimmäkseen indikatiivia (palautan, puhumme) ja imperatiivin 2. persoonaa (palauta, puhukaa). Tekstissä ei ole harvinaisia verbimoduksia, kuten potentiaalia (tehnee, tietänemme) ja vanhahtavia 3. persoonan imperatiivimuotoja (tehköön).
27. Konditionaalia käytetään vain, jos sitä ei voi korvata indikatiivilla ilman, että merkitys selvästi muuttuu. (Hakemus kannattaa jättää, vaikka et olisi saanut vielä opiskelupaikkaa.)
28. Substantiiveilla ei ole useita määritteitä (Sairastuminen voi olla vakava, vaarallinen ja pelottava asia.).
29. Yhteen kuuluvat sanat, kuten verbiliitot ja verbien rektion mukaiset ilmaukset, esitetään peräkkäin tai mahdollisimman lähekkäin (Päätös vaikuttaa ensi kuun alusta alkaen asumistukeesi. > Päätös vaikuttaa asumistukeesi ensi kuun alusta alkaen.).
30. Tekstissä on pääosin lauseita, jotka rakentuvat persoonamuotoisen verbin varaan (Palauta lomake ajoissa. Vastaamme viesteihin maanantaisin.).
31. Tekstissä ei ole substantiivityylisiä ilmauksia (ns. substantiivitauti, esim. Projektin toteutuksen suunnittelu aikataulutetaan.).
32. Tekstissä käytetään suoraa sanajärjestystä (esim. subjekti, predikaatti, objekti). Käänteistä sanajärjestystä käytetään vain silloin, jos tekstin rakenne niin vaatii tai teksti muuten muuttuu monotoniseksi.
33. Predikaatti sijaitsee lauseen alkupuolella.
34. Tekstissä käytetään passiivia vain silloin, kun tekijä ei ole tiedossa tai tekijän mainitseminen ei ole olennaista (Presidentinvaalit järjestetään kuuden vuoden välein. Talo on rakennettu 1920-luvulla.).
35. Lauseessa ei esitetä kaksinkertaista kieltoa (Laskuja ei saa jättää maksamatta.).
36. Virkerakenteet ovat yksinkertaisia. Sivulauseita on pääosin vain yksi.
37. Virkkeissä päälause ja sivulause ovat tekstin etenemisen kannalta loogisessa järjestyksessä.
38. Virkkeissä lauseet on sidottu toisiinsa esimerkiksi konjunktioilla (siksi, mutta, kun) niin, että asioiden väliset suhteet käyvät ilmi tekstistä.
39. Tekstissä ei ole kiilalauseita (Palveluja, joita ovat esimerkiksi asumispalvelut, vammaispalvelut ja vanhuspalvelut, voit hakea tällä lomakkeella.).
40. Virkkeissä ei esitetä monimutkaisia kieltosuhteita (Et saa kurssimerkintää, jos et palauta tehtävää).

Arvioi annettua tekstiä näiden kriteerien pohjalta ja listaa kaikki välttämättömät korjaukset tekstiin. Anna ehdotuksesi aina seuraavassa muodossa. 

# arvio tekstistä #
{yleinen arviosi tekstistä ja sen selkeydestä}

# korjausehdotukset #
{tarkka lista välttämättömistä korjausehdotuksista tekstiin}
"""

editor_system_prompt="""Editori. Olet toimittaja ja kirjoitetun viestinnän kokenut asiantuntija. 
Arvioit selkokielistä tekstiä ja siihen ehdotettujen korjausten välttämättömyyttä, sekä tekstin julkaisukelpoisuutta. Anna lyhyt arviosi ehdotetuista korjauksista. 
Valitse sitten toinen seuraavista vaihtoehdoista:
1. Jos teksti on jo riittävän hyvä ja korjausehdotukset eivät ole tarpeellisia, vastaa 'TERMINATE'
2. Jos korjaukset ovat tarpeellisia ja teksti on kirjoitettava uudelleen, vastaa 'Kirjoitetaan uusi versio tekstistä'

Anna vastauksesi seuraavassa muodossa:

# arvio #
{yleinen arviosi tekstin julkaisukelpoisuudesta}

# päätös #
{valittu vaihtoehto}
"""

fact_checker_prompt = '''Faktantarkastaja. Olet faktantarkastaja, jonka tehtävä on tarkastaa, että selkokielinen versio tekstistä vastaa keskeisten FAKTOJEN osalta alkuperäistä tekstiä. Et ota kantaa luettavuuteen, ainoastaan asiavirheisiin. 
Käyt yksityskohtaisesti kaikki tekstissä esiintyvät faktat läpi ja varmistat, että faktat eivät oleellisesti muutu tai vääristy. Asiat voidaan kertoa eri tavalla ja eri muodoissa, mutta keskeisten faktojen on aina pysyttävä ennallaan (esimerkiksi lukumäärät, tapahtumat ja väitteet).
Kun olet lukenut alkuperäisen ja selkokielisen tekstin huolellisesti läpi, annat palautteesi ja korjausehdotuksesi. Et kirjoita uutta versiota tekstistä, ainoastaan ehdotat kriittisiä korjauksia.

Alkuperäinen teksti:

"{old_text}"

Uusi teksti:

"{new_text}"

Anna vastauksesi seuraavassa muodossa. Älä kirjoita mitään ylimääräistä.

# arvio tekstistä #
{yleinen arviosi tekstin faktasisällöstä}

# korjausehdotukset #
{lista tarvittavista kriittisistä korjauksista uuteen tekstiin}
'''
def term_msg(msg):
    return "TERMINATE" in msg["content"]

def nullify_history(processed_messages):
    processed_messages = [processed_messages[-1]]
    processed_messages[0]['content']=''
    return processed_messages

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
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(input_text)
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
    description="Kriitikko. Selkosuomen asiantuntija. Osaa arvioida tekstin selkeyttä ja antaa tarvittaessa rakentavaa palautetta ja korjausehdotuksia tekstiin.",
    system_message=critic_system_prompt,
    is_termination_msg=None, #lambda msg: "TERMINATE" in msg["content"],
)
editor = autogen.AssistantAgent(
    name="Editori",
    llm_config=gpt4_config_full,
    description="Editori. Viestinnän asiantuntija ja oikolukija, joka lukee selkosuomenksi kirjoitetun tekstin ja arvioijien kommentit ja korjausehdotukset. Päättää pitääkö teksti kirjoittaa uudelleen korjattuna.",
    system_message=editor_system_prompt,
    is_termination_msg=term_msg,
)

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

        groupchat = autogen.GroupChat(
            agents=[user_proxy, writer, critic,fact_checker,editor], messages=[], allow_repeat_speaker=False, max_round=15,
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
        while current_round<4:
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
