from gtts import gTTS
import os


myText = input("what do you want me to say: ")
heathens = "all my friends are heathens, take it slow wait for them to ask you who you know"
song1 = "its been a long, see you again"

language = "en"
if myText == str("see you again"):
    output = gTTS(text=song1, lang=language, slow=False)
    output.save("output.mp3")
    os.system("start output.mp3")


elif myText == str("heathens"):
    output = gTTS(text=heathens, lang=language, slow=False)
    output.save("output.mp3")
    os.system("start output.mp3")



else:
    output = gTTS(text=myText, lang=language, slow=False)
    output.save("output.mp3")
    os.system("start output.mp3")

