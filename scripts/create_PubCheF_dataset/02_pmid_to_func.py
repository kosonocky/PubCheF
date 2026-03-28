import os
import asyncio
import aiohttp
from lxml import etree
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
from pathlib import Path
import backoff
import openai
import time
import multiprocessing as mp
import argparse

# Set via environment variables: ENTREZ_EMAIL, ENTREZ_API_KEY, OPENAI_API_KEY
ENTREZ_EMAIL = os.environ.get('ENTREZ_EMAIL', 'example@domain.com')
ENTREZ_API_KEY = os.environ.get('ENTREZ_API_KEY', '')

# Without an API key NCBI allows 3 req/s; with one, 10 req/s
_ENTREZ_MAX_RATE = 10 if ENTREZ_API_KEY else 3
_ENTREZ_DELAY   = 1.0 / _ENTREZ_MAX_RATE
entrez_semaphore = asyncio.Semaphore(_ENTREZ_MAX_RATE)

class EntrezRateLimitError(Exception):
    pass

@backoff.on_exception(backoff.expo,
                      (aiohttp.ClientError, EntrezRateLimitError),
                      max_tries=8,
                      )
async def fetch_details(pmid):
    """
    Fetches title and abstract from PubMed using Entrez
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'id': pmid,
        'rettype': 'medline',
        'retmode': 'xml',
    }
    if ENTREZ_API_KEY:
        params['api_key'] = ENTREZ_API_KEY
    async with entrez_semaphore:
        await asyncio.sleep(_ENTREZ_DELAY)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    raise EntrezRateLimitError(f"Rate limited (429) for PMID {pmid}")
                if response.status == 200:
                    data = await response.text()
                    root = etree.fromstring(data.encode('utf-8'))
                    title = root.xpath('.//ArticleTitle/text()')
                    abstract_texts = root.xpath('.//AbstractText/text()')

                    title = title[0] if title else None
                    abstract = " ".join(abstract_texts) if abstract_texts else None

                    return title, abstract, response.status
                else:
                    return None, None, response.status

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

async def get_chat_response(client, system_message: str, user_request: str, model: str = "gpt-4o-mini", seed: int = 42, temperature: float = 0):
    """
    Get a chat response from OpenAI chat model.
    """
    try:
        if len(system_message) == 0 or len(user_request) == 0 or len(model) == 0:
            print("ERROR: MISSING PROMPT OR API KEY OR MODEL")
            return None
        response = completions_with_backoff(
            client,
            model=model,
            messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_request},
                ],
            seed=seed,
            stream=False,
            temperature=temperature,
            )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def process_single_pmid(client, pmid: list, gpt_model: str, system_message: str, user_request_partial: str, seed: int, temperature: float):
    """
    Process a single PMID. Fetches title and abstract from PubMed, then uses ChatGPT with prompt to generate a response.
    """
    title, abstract, entrez_status_code = await fetch_details(str(pmid))
    user_request = f"{user_request_partial}\n\nTitle:\n{title}\n\nAbstract:\n{abstract}"

    chat_response = await get_chat_response(
        client,
        system_message=system_message,
        user_request=user_request,
        temperature=temperature,
        model=gpt_model,
        seed=seed,
    )
    return {"pmid": pmid, "entrez_status_code": entrez_status_code, "chat_response": chat_response}


async def process_pmids(client, pmids: list, gpt_model: str, system_message: str, user_request_partial: str, seed: int, temperature: float):
    """
    Process a list of PMIDs. Fetches title and abstract from PubMed, then uses ChatGPT with prompt to generate a response.
    This is done in parallel using asyncio. It creates a task for each PMID, awaits the results, gathers all at once, and returns a dataframe.
    """
    tasks = [process_single_pmid(client, pmid, gpt_model, system_message, user_request_partial, seed, temperature) for pmid in pmids]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


async def main(args):
    t_start = time.time()
    t_curr = t_start
    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment
    gpt_model = args.gpt_model
    print(f"INFO: Using {(temperature:=0)} temperature for GPT")
    print(f"INFO: Using seed {(seed:=42)}")
    print(f"INFO: Testing = {(testing:=False)}")
    print(f"INFO: Using {gpt_model} for model")
    data_path = Path(args.pubchem_dir)
    print(f"INFO: Data path is {data_path}")
    assert data_path.exists(), f"ERROR: Data path {data_path} does not exist"
    print(f"INFO: Outputting to {(output_dir:=Path(f'../../data/dataset_creation/bioassay_pmids/{gpt_model}'))}")
    print(f"INFO: Rerunning errors from 'all' directory: {(rerun_errors:=False)}")
    if testing:
        output_dir = Path(f"{output_dir}/test")
    else:
        output_dir = Path(f"{output_dir}/all")
        if rerun_errors:
            output_dir = Path(f"{output_dir}/rerun_errors")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the prompt
    gpt_summ_system_prompt = "You are an organic chemist summarizing scientific literature"
    gpt_summ_user_prompt = r"Return a set of a few (1-5) 1-3 word descriptors that best describe the chemical or pharmacological function(s) of the molecule described by the given article. Be concise, as specific as possible, but not necessarily comprehensive (choose a small number of great descriptors). For example, prefer 'Dihydrofolate Reductase Inhibitor' over 'Enzyme Inhibitor'. Follow the syntax '{descriptor_1} / {descriptor_2} / {etc}', writing 'NA' if nothing is provided. DO NOT BREAK THIS SYNTAX. The following is the article info:"
    with open(f"{output_dir}/prompt.txt", "w") as f:
        f.write(gpt_summ_system_prompt + "\n\n" + gpt_summ_user_prompt)

    # Load the PMIDs
    if testing:
        with open(f"finetune_chatgpt/test_pmids.txt", "r") as f:
            pmids = f.read().split("\n")
    else:
        if rerun_errors:
            with open(f"{output_dir}/../../pmids_failed_v1.txt", "r") as f:
                pmids = f.read().split("\n")
        else:
            with open(f"{data_path}/bioassay_pmids_max_1k_rand.txt", "r") as f:
                pmids = [p.strip() for p in f.read().split("\n") if p.strip()]

    if args.n_pmids is not None:
        pmids = pmids[:args.n_pmids]
        print(f"INFO: Limiting to first {args.n_pmids} PMIDs (demo mode)")

    print(f"Beginning ChatGPT API calls... Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
    start_idx = 0
    chunk_files = []
    # Process the PMIDs in chunks of 1000
    for i in range(start_idx, len(pmids), 1000):
        if len(pmids) - i < 1000:
            lim = len(pmids) - i
        else:
            lim = 1000
        print(f"Processing PMIDs {i} to {i+lim}... Time: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds\n")
        pmids_subset = pmids[i:i+lim]
        df = await process_pmids(client, pmids_subset, gpt_model, gpt_summ_system_prompt, gpt_summ_user_prompt, seed, temperature)
        chunk_path = f"{output_dir}/pmids_{i}.csv"
        df.to_csv(chunk_path, index=False)
        chunk_files.append(chunk_path)

    # Consolidate all chunks into a single file for downstream scripts
    combined = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    combined.to_csv(f"{output_dir}/../pmid_func_complete.csv", index=False)
    print(f"INFO: Consolidated {len(chunk_files)} chunk(s) into pmid_func_complete.csv")

    print("Complete. Thank you for your patience.")
    print(f"Total time elapsed: {time.time() - t_start}")

if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts and annotate with GPT")
    parser.add_argument("--gpt_model", default="gpt-4o-mini",
                        help="OpenAI chat model to use (e.g. 'gpt-4o-mini', 'gpt-4o', or a fine-tuned model ID)")
    parser.add_argument("--pubchem_dir", default=f"../../data/pubchem/{time.strftime('%Y%m%d')}",
                        help="Path to the PubChem data directory produced by step 01")
    parser.add_argument("--n_pmids", type=int, default=None,
                        help="Limit to the first N PMIDs (useful for demos/testing)")
    args = parser.parse_args()
    asyncio.run(main(args))