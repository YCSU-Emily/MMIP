# Assignment 3 — Integrated Multimodal Customer-Service Chatbot

An integrated multi-turn, multimodal customer-service chatbot combining intelligent routing, contextual conversation management, image understanding, and prompt-injection protection.

This project integrates the previous assignments into a unified customer-service application for 3C products and after-sales support.

---

## 1. Project Overview

The goal of this assignment is to build an integrated chatbot that can understand user requests, select an appropriate processing path, maintain conversation context, and handle both text and image inputs.

### Main Features

- Multi-turn text conversations.
- Multimodal input with image uploads.
- Intelligent Fast Path and Deep Path routing.
- Context-aware follow-up question handling.
- Topic-shift detection.
- Conversation history management.
- Text-based prompt-injection detection.
- Image-based prompt-injection detection through OCR.
- Structured JSON output validation.
- Interactive Streamlit user interface.

## 2. System Architecture

The system processes user input through security inspection, context analysis, routing, response generation, and output validation.

~~~mermaid
flowchart TD
    A["User Input"] --> B{"Input Type"}
    B --> C["Text Input"]
    B --> D["Image Upload"]
    C --> E["Text Security Inspection"]
    D --> F["Image Validation"]
    F --> G["OCR-based Image Security Inspection"]
    E --> H{"Security Check"}
    G --> H
    H -->|Suspicious| I["Block Unsafe Input"]
    H -->|Safe| J["Context Manager"]
    J --> K["Intent and Topic Analysis"]
    K --> L{"Routing Decision"}
    L -->|Fast Path| M["Assignment 1 Chatbot"]
    L -->|Deep Path| N["Local Qwen Model"]
    M --> O["Output Schema Validation"]
    N --> O
    O --> P["Final Response"]
~~~

### Processing Workflow

1. Receive the user's text and optional image.
2. Validate the uploaded image.
3. Inspect text and image content for suspicious instructions.
4. Analyze conversation history and current topic.
5. Select Fast Path or Deep Path.
6. Generate a response.
7. Validate the response against the JSON output schema.
8. Display the result in the Streamlit interface.

## 3. Intelligent Routing: Fast Path and Deep Path

The chatbot uses an intelligent routing mechanism to select an appropriate processing path based on the user's request.

~~~mermaid
flowchart TD
    A["Incoming Message"] --> B["Intent and Context Analysis"]
    B --> C{"Select Processing Path"}
    C -->|Simple Request| D["Fast Path"]
    C -->|Complex or Multimodal Request| E["Deep Path"]
    D --> F["Assignment 1 Chatbot"]
    E --> G["Local Qwen Vision-Language Model"]
    F --> H["Structured Output"]
    G --> H
    H --> I["Final Chatbot Response"]
~~~

### Fast Path

Fast Path handles relatively simple customer-service requests.

Examples:

- Product information.
- Basic troubleshooting.
- Frequently asked questions.
- Simple after-sales inquiries.

It reuses the chatbot implementation from Assignment 1.

### Deep Path

Deep Path handles requests that require more complex reasoning or multimodal understanding.

Examples:

- Complex troubleshooting.
- Image-based product inspection.
- Requests that require combining image information with conversation context.

The system loads the locally configured Qwen model when Deep Path is needed.

The model is loaded lazily rather than being initialized immediately when the application starts.

## 4. Multi-turn Conversation and Context Management

The chatbot maintains both complete conversation history and active-topic context.

~~~mermaid
flowchart TD
    A["New User Message"] --> B["Read Conversation History"]
    B --> C["Analyze Topic Relationship"]
    C --> D{"Related to Active Topic?"}
    D -->|Yes| E["Keep Existing Topic Context"]
    D -->|No| F["Clear Previous Active Context"]
    F --> G["Preserve Complete Conversation History"]
    E --> H["Update Active Context"]
    G --> H
    H --> I["Continue Routing and Response"]
~~~

### Conversation History

The system stores the conversation history so previous messages remain available during the session.

### Active Context

Only messages belonging to the current topic are included in the active model context.

This prevents unrelated previous conversations from interfering with the current request.

### Topic-shift Detection

When a user changes to an unrelated topic:

1. The system detects the topic change.
2. The previous active-topic context is cleared.
3. The complete conversation history is preserved.
4. The chatbot acknowledges the topic transition.
5. The new request is processed using the updated context.

This design improves contextual relevance and reduces interference between unrelated requests.

## 5. Multimodal Input Processing

The chatbot supports both text and image input.

Users can upload JPG, JPEG, PNG, or WEBP images through the Streamlit interface.

Image input is validated before normal chatbot processing.

The image may then be passed to the Deep Path for visual understanding, depending on the routing decision.

The application also performs OCR-based inspection of visible image text to identify known prompt-injection patterns.

## 6. Prompt-Injection Security

Security inspection is performed before normal routing and model inference.

~~~mermaid
flowchart TD
    A["Incoming Text and Image"] --> B["Inspect Text"]
    A --> C["Validate Image"]
    C --> D["OCR Image Text"]
    B --> E["Check Suspicious Patterns"]
    D --> E
    E --> F{"Suspicious Content?"}
    F -->|Yes| G["Block Request"]
    G --> H["Security Review Response"]
    F -->|No| I["Continue Normal Processing"]
~~~

### 6.1 Text-based Prompt-Injection Detection

The text security module checks incoming messages against predefined suspicious patterns.

Examples include instructions attempting to:

- Ignore previous instructions.
- Reveal the system prompt.
- Override system or developer instructions.
- Disclose hidden configuration or internal information.

Both English and Traditional Chinese test cases are included.

### 6.2 Image-based Prompt-Injection Detection

The image security module uses OCR to extract visible text from uploaded images.

The extracted text is inspected using the text security detection mechanism.

This helps identify malicious instructions embedded in screenshots or other uploaded images.

### 6.3 Security Handling

When suspicious content is detected:

- The request is blocked before normal routing.
- The normal response-generation path is not executed.
- The result is marked as a security review.
- The system can indicate that human assistance is required.

The OCR implementation is dependent on successful text extraction. OCR failures are not equivalent to a successful security inspection, and images containing non-extractable or visually concealed instructions may not be detected.

## 7. Structured Output Validation

The chatbot uses `config/output_schema.json` to validate generated responses.

The schema follows JSON Schema Draft 2020-12.

The output contains six required top-level fields:

| Field | Description |
|---|---|
| `intent` | The detected customer-service intent |
| `entities` | Extracted entities from the conversation |
| `image_analysis` | Image-related analysis information |
| `conversation_state` | Current conversation state |
| `routing` | Selected processing path and routing information |
| `response` | Final response displayed to the user |

The schema also uses `additionalProperties: false` to prevent unexpected top-level fields.

## 8. User Interface and User Experience

The application is implemented with Streamlit.

Main interface features:

- Chat-style conversation layout.
- Text input for customer inquiries.
- Image upload support.
- Sidebar feature descriptions.
- Session reset button.
- Loading indicators during processing.
- Expandable diagnostic information.
- Display of routing and security-related information.

### UX Improvements

The application includes several improvements to make interactions easier to understand:

1. **Multi-turn continuity:** Users can ask follow-up questions without repeating all previous information.
2. **Topic transition feedback:** The chatbot explicitly acknowledges when the user changes to an unrelated topic.
3. **Session reset:** Users can start a new conversation when needed.
4. **Processing feedback:** Loading indicators make model processing easier to follow.
5. **Diagnostic visibility:** Expandable information helps inspect routing and security decisions.

## 9. Evaluation and Experimental Results

The project includes separate evaluation scripts for text-based and image-based prompt-injection detection.

### 9.1 Text Security Evaluation

Evaluation file:

`results/text_security_evaluation.json`

| Metric | Result |
|---|---:|
| Total test cases | 18 |
| True Positive (TP) | 13 |
| False Negative (FN) | 0 |
| False Positive (FP) | 0 |
| True Negative (TN) | 5 |
| Attack Recall | 100% |
| False Positive Rate | 0% |
| Accuracy | 100% |

The test set includes English and Traditional Chinese prompt-injection examples, as well as benign customer-service questions.

### 9.2 Image Security Evaluation

Evaluation file:

`results/image_security_evaluation.json`

| Metric | Result |
|---|---:|
| Total test cases | 3 |
| True Positive (TP) | 2 |
| False Negative (FN) | 0 |
| False Positive (FP) | 0 |
| True Negative (TN) | 1 |
| Attack Recall | 100% |
| False Positive Rate | 0% |
| Accuracy | 100% |

The test set includes English and Chinese image-based attack examples and a normal product image.

### 9.3 Interpretation and Limitations

Both evaluation scripts run with normal model inference disabled.

Therefore, these results evaluate the implemented detection mechanisms and their test cases. They do not demonstrate that the complete chatbot is immune to all prompt-injection attacks.

The image evaluation dataset is particularly small. More diverse images, OCR failure cases, indirect instructions, and adversarial examples are needed for a broader security assessment.

## 10. Screenshots and Demonstrations

The following screenshots document the main functions and evaluation results.

### 10.1 Main Chatbot Interface

![Main chatbot interface](docs/screenshots/01_chat_interface.png)

### 10.2 Multi-turn Conversation

![Multi-turn conversation](docs/screenshots/02_multiturn_conversation.png)

### 10.3 Topic-shift Detection

![Topic-shift detection](docs/screenshots/03_topic_shift.png)

### 10.4 Deep Path Processing

![Deep Path processing](docs/screenshots/04_deep_path.png)

### 10.5 Text Prompt-injection Detection

![Text prompt-injection detection](docs/screenshots/05_text_injection.png)

### 10.6 Image Prompt-injection Detection — Chinese

![Chinese image prompt-injection detection](docs/screenshots/06_image_injection_chinese.png)

### 10.7 Normal Image Accepted

![Normal image accepted](docs/screenshots/07_normal_image_allowed.png)

### 10.8 Session Reset

![Session reset](docs/screenshots/08_session_reset.png)

### 10.9 Unit Test Results

![Unit test results](docs/screenshots/09_unit_tests_passed.png)

### 10.10 Text Security Evaluation Results

![Text security evaluation results](docs/screenshots/10_text_security_evaluation.png)

### 10.11 Image Security Evaluation Results

![Image security evaluation results](docs/screenshots/11_image_security_evaluation.png)

## 11. Project Structure

~~~text
assignment3_integrated_chatbot/
├── app.py
├── src/
│   ├── integrated_chatbot.py
│   ├── context_manager.py
│   ├── security.py
│   └── image_security.py
├── prompts/
│   └── routing_prompt_reference.py
├── config/
│   └── output_schema.json
├── evaluation/
│   ├── evaluate_text_security.py
│   └── evaluate_image_security.py
├── results/
│   ├── text_security_evaluation.json
│   └── image_security_evaluation.json
├── docs/
│   └── screenshots/
├── tests/
└── README.md
~~~

## 12. Environment and Execution

### Environment

The project uses the `mmipQuiz` Conda environment.

Activate the environment:

~~~bash
conda activate mmipQuiz
~~~

### Start the Application

From the project directory, run:

~~~bash
streamlit run app.py
~~~

Open the local URL displayed in the terminal to access the chatbot.

### Run Text Security Evaluation

~~~bash
python evaluation/evaluate_text_security.py
~~~

### Run Image Security Evaluation

~~~bash
python evaluation/evaluate_image_security.py
~~~

## 13. Assignment Requirement Mapping

| Assignment Requirement | Implementation | Evidence |
|---|---|---|
| Routing and multi-turn multimodal chatbot (5 points) | Fast Path, Deep Path, text/image input, context management | Sections 2–5; Screenshots 1–4 |
| Reasonable chatbot interface (5 points) | Streamlit chat UI, upload, reset, loading feedback | Section 8; Screenshots 1 and 8 |
| UX improvement and effectiveness (5 points) | Topic-shift feedback and active-context management | Section 4; Screenshot 3 |
| Handle sudden unrelated questions (10 points) | Topic-shift detection, active-context reset, preserved history | Section 4; Screenshot 3 |
| Text and image prompt-injection defense (10 points) | Text pattern inspection and OCR-based image inspection | Section 6; Screenshots 5–7 |

## 14. Conclusion

This assignment integrates routing, multi-turn conversation management, multimodal input, user-interface design, and prompt-injection detection into a single customer-service chatbot.

The system distinguishes between simple and complex requests through Fast Path and Deep Path routing. It maintains complete conversation history while isolating the active topic context, helping reduce interference when users change subjects.

In addition, text and OCR-based image security checks are performed before normal routing and model inference.

The evaluation results demonstrate that the implemented detectors successfully identify the included test cases. Further evaluation with larger and more diverse datasets is required to assess generalization and robustness.

Overall, the project demonstrates an integrated multimodal customer-service workflow with contextual routing, improved interaction experience, and basic defense against text- and image-based prompt injection.
