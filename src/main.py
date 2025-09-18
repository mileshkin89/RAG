import json

from rag_pipline import get_context, get_llm_response


def main():
    query = input("Input your query: ")

    context = get_context(query)

    print("CONTEXT:")
    print(json.dumps(context, indent=4, ensure_ascii=False))

    response = get_llm_response(query, context)

    print("YOUR QUERY:")
    print(query)
    print("RESPONSE:")
    print(response)




if __name__ == "__main__":
    main()


#
# "What's the spiciest dish you know? What ingredients does it contain?"
# "Write a recipe for a vegetarian salad with avocado"