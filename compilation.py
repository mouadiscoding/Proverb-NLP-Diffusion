from ply import lex, yacc
from openai import OpenAI
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
import speech_recognition as sr
import os



global client
client = OpenAI(api_key='OPENAI_API_KEY')
st.header('Projet Compilation', divider='green')
st.header('', divider='green')
st.title("Analyseur lexical:")

#analyseur lexical : 

tokens = ['SE','NE','PAS','QUOTE','SUJET','PHRASE','PREPOSITION','COMPLEMENT','PRONOM','ARTICLE','VERBE','ADVERBE','VIRGULE','POINT','APOSTROPHE','CONJONCTION','ADJECTIF','SES'] #18
st.subheader("tokens utilisés : ")

tokens_text = ' , '.join(tokens)
st.write(tokens_text)
t_SE = r'se|Se|s\''  #3
t_NE = r'ne|Ne|n\'' #3
t_PAS = r'pas|Pas' #2
t_VIRGULE  = r'\,' #1
t_POINT = r'\.' #1
t_QUOTE = r'\"' #1
t_VERBE = r'manger|passe|sont|restent|reste|insistons|dancer|passe|vaut|est|devient|dancent|fait|veut|peut|prévenir|guerir|Donner|donner|reprendre|voler|casser|couche|lève|forgeant|vient|moquer|faut|arrive|mange|suivent|ressemblent|mords|nourrit|change|gagne|reste|donne|avoir|arrive|es|occupé|faire|attendre|apprendre|compliquer|aiment|font|envolent|oublient|oublie|risque|as|vient|travailler|Parlons|parlons|travaille|boire'  #58
t_SUJET = r'pere|fils|père|on|chat|souris|il|Il|imbéciles|voisin|vengeance|visage|jours|On|vie|amour|personnes|paroles|écrits|prix|qualité|hommes|temps|mort' #24
t_PRONOM = r'C\'|c\'|que|qu\'|qui|Qui|il|la|Il|te|ce|tu|nous' #13
t_ARTICLE = r'L\'|en|une|un|des|les|la|le|l\'|d\'|là|du|de|La|Les|Le' #16
t_ADVERBE = r'tard|quand|Mieux|mieux|jamais|toujours|vraiment|peu' #8
t_COMPLEMENT = r'enfants|avis|omelettes|oeufs|chiens|puces|forgeron|temps|patience|tout|bout|peine|matin|plat|froid|miroir|coeur|main|équipe|peu|parfum|roses|ambitions|moyens|autres|plans|nouvelles|pluie|orage|mal|rien|lime|bruit' #34
t_PREPOSITION = r'sans|Sans|avec|Avec|à|pendant|sous|mais|pour' #9
t_ADJECTIF = r'vôtre|ses|bonnes|simple|grands|bien|Bien' #7
t_CONJONCTION = r'et|car' #2 
t_ignore = ' \t' #1

#183

#ARTICLE SUJET NE VERBE PAS COMPLEMENT VIRGULE ARTICLE SUJET PRONOM NE VERBE PAS VERBE COMPLEMENT POINT
#L'amour ne fait pas mal, les personnes qui n'aiment pas font mal.
#ARTICLE SUJET VERBE ADVERBE ADJECTIF VIRGULE PREPOSITION PRONOM VERBE PREPOSITION ARTICLE VERBE POINT 
#La vie est vraiment simple, mais nous insistons pour la compliquer.
#ARTICLE SUJET NE VERBE PAS ARTICLE VERBE PRONOM ARTICLE COMPLEMENT VERBE VIRGULE PRONOM VERBE ARTICLE VERBE PREPOSITION VERBE PREPOSITION ARTICLE COMPLEMENT
#La vie n'est pas d'attendre que l'orage passe, c'est d'apprendre à danser sous la pluie.
#PAS ARTICLE COMPLEMENT VIRGULE ADJECTIF COMPLEMENT POINT
#Pas de nouvelles, bonnes nouvelles.
#ARTICLE SUJET VERBE PRONOM PRONOM VERBE PREPOSITION PRONOM PRONOM VERBE VERBE PREPOSITION VERBE ARTICLE COMPLEMENT COMPLEMENT POINT
#La vie est ce qui arrive pendant que tu es occupé à faire d'autres plans.
#ARTICLE VERBE ADVERBE ARTICLE COMPLEMENT ARTICLE COMPLEMENT PREPOSITION ARTICLE COMPLEMENT PRONOM VERBE ARTICLE COMPLEMENT POINT
#Il reste toujours un peu de parfum à la main qui donne des roses.
#ARTICLE VERBE ADVERBE VERBE ARTICLE COMPLEMENT ARTICLE ADJECTIF COMPLEMENT POINT
#Il faut toujours avoir les moyens de ses ambitions.
#SUJET NE VERBE PAS ARTICLE COMPLEMENT PRONOM VERBE POINT
#On ne change pas une équipe qui gagne.
#NE VERBE PAS ARTICLE COMPLEMENT PRONOM PRONOM VERBE POINT
#Ne mords pas la main qui te nourrit.
#ARTICLE SUJET SE VERBE CONJONCTION NE SE VERBE PAS POINT
#Les jours se suivent et ne se ressemblent pas.
#ARTICLE SUJET VERBE ARTICLE COMPLEMENT ARTICLE COMPLEMENT POINT
#'Le visage est le miroir du coeur.'
#PREPOSITION ARTICLE COMPLEMENT CONJONCTION ARTICLE ARTICLE COMPLEMENT VIRGULE SUJET VERBE PREPOSITION COMPLEMENT ARTICLE COMPLEMENT POINT
#Avec du temps et de la patience, on vient à bout de tout.
#ARTICLE SUJET VERBE ARTICLE COMPLEMENT PRONOM SE VERBE COMPLEMENT POINT
#La vengeance est un plat qui se mange froid.


def t_error(t):
    print("-----------ANALYSE LEXICAL--------------")
    print(f"Illegal character {t.value[0]} at column {t.lexpos}")
    st.text("❌ Analyse léxicale incorrecte !!! ❌")
    st.text(f"❌ Illegal character '{t.value[0]}' at column {t.lexpos} ❌")
    t.lexer.skip(1)
lexer = lex.lex()
st.markdown("---")
#analyseur syntaxique :

def p_quote(p):
    """ quote  : phrase
                | phrase VIRGULE phrase
                | phrase VIRGULE phrase POINT
                | phrase POINT"""
    print("analyse semantique correct ")
    st.text("✅ Analyse sémantique correcte !!! ✅ ")

    concatenated_phrase = ""

    for i in range(1, len(p)):
        concatenated_phrase += p[i] + " "  # Add a space between tokens for clarity
    
    # Remove trailing space if necessary
    concatenated_phrase = concatenated_phrase.strip()
    
    # Set p[0] to the concatenated phrases
    p[0] = concatenated_phrase

    # Display options to generate story
    if st.button("Générer une histoire"):
        generate_story_from_quote(p[0])
        print("...")
    if st.button("Générer une image"):
        generate_image_from_quote(p[0])
        print("...")



def p_phrase(p):
    """
    phrase :  ADVERBE VERBE ADVERBE PRONOM ADVERBE 
            | PRONOM VERBE ARTICLE VERBE PRONOM SUJET VERBE COMPLEMENT
            | PRONOM NE VERBE PAS ADVERBE SUJET VERBE NE VERBE PAS ADVERBE SUJET VERBE 
            | ADVERBE VERBE VERBE PRONOM VERBE
            | SUJET NE VERBE PAS ARTICLE COMPLEMENT PREPOSITION VERBE ARTICLE COMPLEMENT 
            | PRONOM VERBE PREPOSITION ARTICLE COMPLEMENT SE VERBE PREPOSITION ARTICLE COMPLEMENT 
            | PREPOSITION ARTICLE COMPLEMENT CONJONCTION ARTICLE ARTICLE COMPLEMENT
            | SUJET VERBE PREPOSITION COMPLEMENT ARTICLE COMPLEMENT 
            | ARTICLE SUJET VERBE ARTICLE COMPLEMENT ARTICLE COMPLEMENT 
            | ARTICLE SUJET VERBE ARTICLE COMPLEMENT PRONOM SE VERBE COMPLEMENT 
            | ARTICLE SUJET SE VERBE CONJONCTION NE SE VERBE PAS 
            | NE VERBE PAS ARTICLE COMPLEMENT PRONOM PRONOM VERBE 
            | SUJET NE VERBE PAS ARTICLE COMPLEMENT PRONOM VERBE 
            | ARTICLE SUJET VERBE PRONOM PRONOM VERBE PREPOSITION PRONOM PRONOM VERBE VERBE PREPOSITION VERBE ARTICLE COMPLEMENT COMPLEMENT 
            | PAS ARTICLE COMPLEMENT 
            | ADJECTIF COMPLEMENT 
            | ARTICLE SUJET NE VERBE PAS ARTICLE VERBE PRONOM ARTICLE COMPLEMENT VERBE 
            | PRONOM VERBE ARTICLE VERBE PREPOSITION VERBE PREPOSITION ARTICLE COMPLEMENT 
            | ARTICLE SUJET VERBE ADVERBE ADJECTIF 
            | PREPOSITION PRONOM VERBE PREPOSITION ARTICLE VERBE 
            | ARTICLE SUJET NE VERBE PAS COMPLEMENT 
            | ARTICLE SUJET PRONOM NE VERBE PAS VERBE COMPLEMENT 
            | ARTICLE SUJET VERBE
            | ARTICLE SUJET SE VERBE
            | ARTICLE SUJET VERBE ARTICLE ADJECTIF COMPLEMENT
            | PRONOM NE VERBE COMPLEMENT NE VERBE COMPLEMENT
            | ARTICLE SUJET VERBE CONJONCTION ARTICLE SUJET VERBE
            | ARTICLE SUJET VERBE ARTICLE COMPLEMENT PRONOM VERBE PREPOSITION COMPLEMENT
            | VERBE ADVERBE
            | PREPOSITION VERBE ADJECTIF
            | ADJECTIF VERBE CONJONCTION ADJECTIF VERBE VERBE ADJECTIF VERBE"""

    concatenated_phrase = ""

    for i in range(1, len(p)):
        concatenated_phrase += p[i] + " "  # Add a space between tokens for clarity
    
    # Remove trailing space if necessary
    concatenated_phrase = concatenated_phrase.strip()
    
    # Set p[0] to the concatenated phrase
    p[0] = concatenated_phrase
    


def p_error(p):
    print("---------ANALYSE SYNTAXIQUE----------")
    print("Syntax error in input!")
    st.text("❌ Analyse sémantique incorrecte !!! ❌")

def find_column(input_text, token):
    last_cr = input_text.rfind('\n', 0, token.lexpos)
    if last_cr < 0:
        last_cr = 0
    return token.lexpos - last_cr + 1

parser = yacc.yacc()


def generate_story_from_quote(quote):
    with st.spinner('Generating Story... '):

        prompt = f"donnez moi une petite histoire de nuit qui ne depace pas 2 lignes pour les enfants a partir du proverbe suivant : {quote}"

        # Make a request to the OpenAI GPT-3 API
        response = client.completions.create(
            model="text-davinci-003",  # Specify the GPT-3 model
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            n=1
        )
        # Extract the generated text from the API response
        generated_text = response.choices[0].text
        generated_text = generated_text.lstrip('\n')
        st.subheader(f"Histoire généré du proverbe : {quote}")
        st.write(generated_text)

        if generated_text:
            speak_quote(generated_text)

def speak_quote(quote):

    # Using gTTS to generate audio from text
    tts = gTTS(quote, lang='fr', slow=False)
    
    # Save the generated audio to a temporary file
    temp_file_path = "temp_audio.mp3"
    tts.save(temp_file_path)

    # Display audio player in Streamlit
    st.audio(temp_file_path, format='audio/mp3')

    # Remove the temporary file
    os.remove(temp_file_path)


def generate_image_from_quote(quote):
    with st.spinner('Generating image...'):
        
        prompt = f'Create a sci-fi image inspired by the following French quote: {quote}.'
        pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]

        st.image(image)


def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio_data = recognizer.listen(source, timeout=5)
        
    try:
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        st.write(f"Texte reconnu : {text}")
        # Pass the recognized text to your parser here
        parser.parse(text)
    except sr.UnknownValueError:
        st.write("Désolé, je n'ai pas compris l'audio.")
    except sr.RequestError:
        st.write("Désolé, une erreur s'est produite lors de la demande.")


def main():
    st.title("Analyseur de proverbes ")

    if st.button("Commencer la reconnaissance vocale"):
        recognize_speech()

    if 'selected_quote' not in st.session_state:
        st.session_state.selected_quote = None
    # Display sidebar with list of quotes
    st.sidebar.title("Proverbes")
    quotes = [
    'Mieux vaut tard que jamais.',
    'C\'est en forgeant qu\'on devient forgeron.', 
    'Qui ne fait pas quand il peut ne fait pas quand il veut.',
    'Mieux vaut prévenir que guerir.',
    'On ne fait pas d\'omelettes sans casser d\'oeufs.',
    'Qui couche avec des chiens se lève avec des puces.',
    'Avec du temps et de la patience, on vient à bout de tout.',
    'La vengeance est un plat qui se mange froid.',
    'Le visage est le miroir du coeur.',
    'La vie est ce qui arrive pendant que tu es occupé à faire d\'autres plans.',
    'Pas de nouvelles, bonnes nouvelles.',
    'On ne change pas une équipe qui gagne.',
    'Les jours se suivent et ne se ressemblent pas.',
    'La vie est vraiment simple, mais nous insistons pour la compliquer.',
    'L\'amour ne fait pas mal, les personnes qui n\'aiment pas font mal.',
    'Les paroles s\'envolent, les écrits restent.',
    'Le prix s\'oublie, la qualité reste.',
    'Les hommes sont de grands enfants.',
    'Qui ne risque rien n\'as rien.',
    'Bien boire et bien manger font bien travailler.'
    ]

    for quote in quotes:
        if st.sidebar.button(quote):
            st.session_state.selected_quote = quote
    
    # selected_quote = st.session_state.selected_quote

    selected_quote = st.text_input("Veuillez Entrer Votre Proverbe:", st.session_state.selected_quote)
    

    
    # Display options to generate story
    if selected_quote:
        st.subheader(f"Analyse du proverbe : {selected_quote}")
        parser.parse(selected_quote)  # Assuming you have a valid parser
        st.markdown("---")
        
        
if __name__ == "__main__":
    main()