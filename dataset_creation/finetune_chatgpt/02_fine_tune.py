import os
import time
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# Upload training file
print("Uploading training file...")
upload_response = openai.files.create(
    file=open("train_pmids_refined_examples.json", "rb"),
    purpose='fine-tune',
)
training_file_id = upload_response.id
print(f"Uploaded file: {training_file_id}")

# Start fine-tuning job
print("Starting fine-tuning job...")
job = openai.fine_tuning.jobs.create(
    training_file=training_file_id,
    model="gpt-4o-mini-2024-07-18",
    suffix="pmids_refined",
)
job_id = job.id
print(f"Job ID: {job_id}")

# Poll until complete
print("Waiting for fine-tuning to complete (this may take a while)...")
while True:
    job = openai.fine_tuning.jobs.retrieve(job_id)
    status = job.status
    print(f"  Status: {status}")
    if status in ("succeeded", "failed", "cancelled"):
        break
    time.sleep(60)

if status == "succeeded":
    fine_tuned_model = job.fine_tuned_model
    print(f"\nFine-tuning complete!")
    print(f"Fine-tuned model name: {fine_tuned_model}")
    print(f"\nUse this model name in your config/inference scripts:")
    print(f"  {fine_tuned_model}")
else:
    print(f"\nFine-tuning ended with status: {status}")
    print(f"Error: {job.error}")
