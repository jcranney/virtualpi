#
#NB: The following environment variables need to be set:
# export OPENAI_API_KEY="sk-M...M"
# export SLACK_APP_TOKEN="xapp-1...d"
# export SLACK_BOT_TOKEN="xoxb-2...C"
#See the README for more information:
#https://github.com/davidbrodrick/virtualpi/blob/main/README.md
#
#David Brodrick, 2023

import sys, os, pickle, glob
from slack_bolt import App
from tqdm import tqdm
import numpy as np
from openai import OpenAI
from tika import parser
from slack_bolt.adapter.socket_mode import SocketModeHandler

embed_mode = "local" # "local" or "openai"

llm_client = OpenAI()
llm_model = "gpt-4o"

if embed_mode == "local":
    embed_client = OpenAI(
        base_url="http://localhost:8002/v1",
        api_key = "sk-no-key-required"
    )
    embed_model = "llama"
elif embed_mode == "openai":
    embed_client = llm_client
    embed_model = "text-embedding-3-small"
else:
    raise ValueError(f"embed_mode: `{embed_mode}` not supported. Choose either `local` or `openai`")

PAPERDIR=sys.argv[1]

def chunk(text,chunk_size=1000):
    chunks = []
    for i in range(len(text[::chunk_size])):
        chunks.append(text[i*chunk_size:(i+1)*chunk_size])
    return chunks

def embed(chunks,client,*,model=embed_model):
    return [d.embedding for d in client.embeddings.create(input=chunks,model=model).data]

def format_and_filter_embeddings(metadata,chunks,embeddings):
    print(f"{len([t for t in embeddings if len(t)>0]):d}/{len(embeddings):d} used")
    return [{
        "text":chunk,
        "embedding":embedding,
        "metadata":metadata
        } for chunk,embedding in zip(chunks,embeddings)
        if len(embedding)>0]

#We require a directory containing PDFs as an argument
if len(sys.argv)!=2:
    print("Requires the path to a repository of PDFs as an argument.")
    sys.exit(1)
PAPERDIR=sys.argv[1]
if not os.path.exists(PAPERDIR):
    print("The specified directory does not exist.")
    sys.exit(1)

try:
    with open("%s/data.pkl"%PAPERDIR, "rb") as f:
        #Save this state for next time
        print("\nTrying to load file %s/data.pkl."%PAPERDIR)
        data = pickle.load(f)
except FileNotFoundError:
    papers=[]
    filesfound=glob.glob("%s/*"%PAPERDIR)
    for p in filesfound:
        if p.lower().endswith(".pdf"):
            papers.append(p)

    if not papers:
        print("No PDFs were found in the specified directory.")
        sys.exit(1)

    print("Found %d PDFs in %s"%(len(papers),PAPERDIR))
    data = []
    pbar = tqdm(papers,leave=True,desc="")
    for paper in pbar:
        try:
            #Get the base file name to use as the citation
            citation=os.path.split(paper)[-1]
            citation=citation[0:citation.rfind(".")]
            #Embed this doc
            pbar.set_description(f"parseing  {citation:s}")
            raw = parser.from_file(paper)
            pbar.set_description(f"chunking  {citation:s}")
            chunks = chunk(raw["content"])
            pbar.set_description(f"embedding {citation:s}")
            embeddings = embed(chunks,embed_client)
            data += format_and_filter_embeddings(citation,chunks,embeddings)
        except OSError as e: #Exception as e:
            print("Error processing %s: %s"%(p,e))

    with open("%s/data.pkl"%PAPERDIR, "wb") as f:
        #Save this state for next time
        print("\nSaving state to file %s/data.pkl - this may take some time."%PAPERDIR)
        pickle.dump(data, f)

b = np.array([d["embedding"] for d in data])

def query(prompt,*,data=data,embed_client=embed_client,embed_model=embed_model,
          llm_client=llm_client,llm_model=llm_model,
          k=5,answer_length="about 100 words"):
    a = np.array(embed([prompt],embed_client,model=embed_model))
    cos_sim = (a@b.T)[0]/np.linalg.norm(a.flatten())/np.linalg.norm(b,axis=1)
    sim_index = np.argsort(cos_sim)[::-1][:k]
    contexts = "".join([f'### Citation Key:\n{d["metadata"]}\n### Context:{d["text"]}\n\n' for d in np.array(data)[sim_index]])

    messages = [
        {
            "role" : "system",
            "content" : ("Answer in a direct and concise tone. Your audience is an expert, "
                         "so be highly specific. If there are ambiguous terms or acronyms, "
                         "first define them.")
        },
        {
            "role" : "user",
            "content" : (llm_prompt:=("Answer the question below with the context.\n\n"
                         f"## Contexts:\n\n{contexts}\n\n----\n\n"
                         f"## Question: {prompt}\n\nWrite an answer based on the context. "
                         "If the context provides insufficient information and the "
                         "question cannot be directly answered, reply 'I cannot answer.' "
                         "For each part of your answer, indicate which sources most support "
                         "it via a reference at the end of sentences, in the form "
                         "`(CITATION_KEY)`. Only cite text from the context. "
                         "Write in the style of a Wikipedia "
                         "article, with concise sentences and coherent paragraphs. The context "
                         "comes from a variety of sources and is only a summary, so there "
                         "may inaccuracies or ambiguities. If quotes are present and relevant, "
                         "use them in the answer. This answer will go directly onto Wikipedia, "
                         f"so do not add any extraneous information.\n\nAnswer ({answer_length}):"))
        }
    ]
    response = llm_client.chat.completions.create(model=llm_model,
                                   messages=messages)
    return {
        "answer":response.choices[0].message.content,
        "llm_prompt" : llm_prompt,
        "contexts":contexts,
        "messages":messages,
    }

print("starting up")
#Create handle to Slack
app = App(token=os.environ["SLACK_BOT_TOKEN"])

############################################################
#This function is called when a Slack user mentions the bot
@app.event("app_mention")
def event_test(say, body):
    print("received question, working on answer.")
    try:
        #This gets the question text from the user
        user_question=body["event"]["blocks"][0]["elements"][0]["elements"][1]["text"]
        if user_question:
            #Do the paper-qa query and get the answer to the question
            answer = query(user_question, k=30)
            #Print some stuff locally
            print(answer["answer"])
            print("\n\n\n")
            #Send the (minimal) answer to Slack
            say(answer["answer"])
    except Exception as e:
        print("Error: %s"%e)

#Set up the Slack interface to start servicing requests
print("Starting Slack handler - bot is ready to answer your questions!")
SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
