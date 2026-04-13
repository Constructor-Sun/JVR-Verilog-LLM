import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    input_path = "data/training_data/integral/codev_r1_sft_failed_prompts.jsonl"
    output_path = "data/training_data/integral/codev_r1_sft_new.jsonl"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        lines = f_in.readlines()
        for line in lines:
            data = json.loads(line)
            prompt = data["prompt"]
            prompt = re.sub(r'/(think|no_think)\s*$', '', prompt).strip() + " /think"
            data["prompt"] = prompt

            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
                },
                model="qwen/qwen3-235b-a22b-thinking-2507",
                messages=[
                    {
                        "role": "user",
                        "content": "### Hint: Your generated Verilog code should be in ```verilog``` format, otherwise the model will not accept it. " + prompt
                    }
                ],
                max_tokens=1024 * 16,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "thinking_budget": 1024 * 14,
                    "include_reasoning": True,
                    "provider": {
                        "sort": "cost"
                    }
                },
                
                # stream=True,
                # stream_options={
                #     "include_usage": True
                # }
            )

            reasoning = completion.choices[0].message.reasoning
            content = completion.choices[0].message.content
            
            # reasoning = ""  # Full thinking process
            # context = ""  # Full response
            # is_answering = False  # Indicates whether the response phase has started
            # print("\n" + "=" * 20 + "Thinking process" + "=" * 20 + "\n")

            # for chunk in completion:
            #     if not chunk.choices:
            #         print("\nUsage:")
            #         print(chunk.usage)
            #         continue

            #     delta = chunk.choices[0].delta

            #     # Collect only the thinking content
            #     if hasattr(delta, "reasoning") and delta.reasoning is not None:
            #         if not is_answering:
            #             print(delta.reasoning, end="", flush=True)
            #         reasoning += delta.reasoning

            #     # When content is received, start responding
            #     if hasattr(delta, "content") and delta.content:
            #         if not is_answering:
            #             print("\n" + "=" * 20 + "Full response" + "=" * 20 + "\n")
            #             is_answering = True
            #         print(delta.content, end="", flush=True)
            #         content += delta.content

            reasoning = reasoning.strip()
            if not re.match(r'^\s*<think>', reasoning, re.IGNORECASE):
                reasoning = "<think>\n" + reasoning
            data["response"] = reasoning + content

            print("prompt: ", data["prompt"])
            print("response: ", data["response"])

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            exit()

if __name__ == "__main__":
    main()