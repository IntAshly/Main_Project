import google.generativeai as genai

# Configure the Gemini API key
genai.configure(api_key="AIzaSyAiok6VSBtZtoh2bYYMP0uitJw2pc2d3E4")

# Set up the model
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_response(prompt):
    """
    Generates a response from the Gemini model based on the given prompt.
    """
    try:
        # Add a health-related context to the prompt
        health_prompt = f"As a helpful health assistant, respond to the following query: {prompt}"
        response = model.generate_content(health_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    """
    Main function to run the chatbot.
    """
    print("Welcome to the Gemini AI Health Assistant! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = generate_response(user_input)
        print("Gemini AI: ", response)

if __name__ == "__main__":
    main()
