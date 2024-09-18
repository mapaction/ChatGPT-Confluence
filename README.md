## Confluence Chatbot with GPT-3.5-turbo and Tokenizer

This Streamlit application allows users to interact with a chatbot that retrieves information from a selected Confluence space and generates responses using the OpenAI GPT-3.5-turbo model.

### Features

* **Confluence Integration:** Connects to a Confluence instance to fetch page content and metadata.
* **GPT-3.5-turbo Model:** Leverages the power of OpenAI's language model for generating contextually relevant and informative responses.
* **Tokenization:** Efficiently manages text input to stay within the model's token limits.
* **Chat History:** Maintains a conversation history for reference.
* **Space Selection:** Allows users to choose from available Confluence spaces.
* **Relevant Links:** Provides links to relevant Confluence pages alongside chatbot responses.
* **Error Handling and Logging:** Includes error handling and logging for improved robustness.

### Requirements

* Python 3.x
* Streamlit
* tiktoken
* atlassian-python-api
* openai
* numpy
* pandas
* beautifulsoup4
* nltk
* python-dotenv

### Setup

1. **Install dependencies:**

   ```bash
   pip install streamlit tiktoken atlassian-python-api openai numpy pandas beautifulsoup4 nltk python-dotenv
   ```

2. **Set up environment variables:**

   * Create a `.env` file in your project directory.
   * Add the following variables with your actual values:

     ```
     openai-api-key = <your-openai-api-key>
     confluence-url = <your-confluence-url> (e.g., https://your-domain.atlassian.net/wiki)
     confluence-username = <your-confluence-email>
     confluence-api-token = <your-confluence-api-token>
     ```

3. **Run the application:**

   ```bash
   streamlit run your_script_name.py
   ```

### Usage

1. **Select a Confluence space** from the dropdown.
2. **Enter your query** in the text area.
3. **Click "Get Chatbot Response"** to receive an answer based on the Confluence content.
4. **View the response** and relevant links below.
5. **Continue the conversation** by asking more questions.

### Notes

* Ensure you have the necessary permissions to access the Confluence space.
* The chatbot's knowledge is limited to the content within the selected space.
* The application handles token limits to avoid exceeding the model's capacity.
* Refer to the `logging` output for detailed information on the application's behavior.
* Consider customizing the prompt construction and response filtering for more tailored results.
