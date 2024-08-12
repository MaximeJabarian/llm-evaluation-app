from openai import OpenAI
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import pandas as pd

class QAMetrics:
    def __init__(self, api_key, model_name="gpt-4o-mini", temperature=0):
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = SentenceTransformer("bert-base-nli-mean-tokens")
        self.model_name = model_name  # Store the selected model name
        self.temperature = temperature  # Store the selected temperature

    # Function to calculate coherence score
    def calculate_coherence_score(self, question, answer):
        messages = [
            {"role": "system", "content": "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric."},
            {"role": "user", "content": f"Coherence of an answer is measured by how well all the sentences fit together and sound naturally as a whole. Consider the overall quality of the answer when evaluating coherence. Given the question and answer, score the coherence of answer between one to five stars using the following rating scale:\n\
            1 : the answer completely lacks coherence\n\
            2 : the answer mostly lacks coherence\n\
            3 : the answer is partially coherent\n\
            4 : the answer is mostly coherent\n\
            5 : the answer has perfect coherency\n\
            This rating value should always be an integer between 1 and 5 only, no text. So the rating produced should be 1 or 2 or 3 or 4 or 5.\n\
            \n\
            question: What is your favorite indoor activity and why do you enjoy it?\n\
            answer: I like pizza. The sun is shining.\n\
            stars: 1\n\
            \n\
            question: Can you describe your favorite movie without giving away any spoilers?\n\
            answer: It is a science fiction movie. There are dinosaurs. The actors eat cake. People must stop the villain.\n\
            stars: 2\n\
            \n\
            question: What are some benefits of regular exercise?\n\
            answer: Regular exercise improves your mood. A good workout also helps you sleep better. Trees are green.\n\
            stars: 3\n\
            \n\
            question: How do you cope with stress in your daily life?\n\
            answer: I usually go for a walk to clear my head. Listening to music helps me relax as well. Stress is a part of life, but we can manage it through some activities.\n\
            stars: 4\n\
            \n\
            question: What can you tell me about climate change and its effects on the environment?\n\
            answer: Climate change has far-reaching effects on the environment. Rising temperatures result in the melting of polar ice caps, contributing to sea-level rise. Additionally, more frequent and severe weather events, such as hurricanes and heatwaves, can cause disruption to ecosystems and human societies alike.\n\
            stars: 5\n\
            question: {question}\n\
            answer: {answer}\n\
            stars:"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        coherence_score = int(response.choices[0].message.content)
        return coherence_score

    # Function to compute F1 score
    def compute_f1_score(self, ground_truth: pd.Series, answer: pd.Series) -> pd.Series:
        def normalize_text(text) -> str:
            """Lower text and remove punctuation, articles, and extra whitespace."""
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(text))))

        # Normalize the texts
        normalized_answers = answer.apply(normalize_text)
        normalized_ground_truths = ground_truth.apply(normalize_text)

        # Generate embeddings for normalized texts
        ground_truth_emb = self.model.encode(normalized_ground_truths.tolist())
        answer_emb = self.model.encode(normalized_answers.tolist())

        # Compute cosine similarity for each pair
        similarities = [
            cosine_similarity([gt_emb], [ans_emb])[0][0]
            for gt_emb, ans_emb in zip(ground_truth_emb, answer_emb)
        ]

        return pd.Series(similarities).round(2)
    
    # Function to compute Similrity score
    def calculate_similarity_score(self, question, ground_truth, answer):
        messages = [
            {"role": "system", "content": "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric."},
            {"role": "user", "content": f"Equivalence, as a metric, measures the similarity between the predicted answer and the correct answer. If the information and content in the predicted answer is similar or equivalent to the correct answer, then the value of the Equivalence metric should be high, else it should be low. Given the question, correct answer, and predicted answer, determine the value of Equivalence metric using the following rating scale:\n\
            1 : the predicted answer is not at all similar to the correct answer\n\
            2 : the predicted answer is mostly not similar to the correct answer\n\
            3 : the predicted answer is somewhat similar to the correct answer\n\
            4 : the predicted answer is mostly similar to the correct answer\n\
            5 : the predicted answer is completely similar to the correct answer\n\
            This rating value should always be an integer between 1 and 5 only, no text. So the rating produced should be 1 or 2 or 3 or 4 or 5.\n\
            The examples below show the Equivalence score for a question, a correct answer, and a predicted answer.\n\
            question: What is the role of ribosomes?\n\
            correct answer: Ribosomes are cellular structures responsible for protein synthesis. They interpret the genetic information carried by messenger RNA (mRNA) and use it to assemble amino acids into proteins.\n\
            predicted answer: Ribosomes participate in carbohydrate breakdown by removing nutrients from complex sugar molecules.\n\
            stars: 1\n\
            question: Why did the Titanic sink?\n\
            correct answer: The Titanic sank after it struck an iceberg during its maiden voyage in 1912. The impact caused the ship's hull to breach, allowing water to flood into the vessel. The ship's design, lifeboat shortage, and lack of timely rescue efforts contributed to the tragic loss of life.\n\
            predicted answer: The sinking of the Titanic was a result of a large iceberg collision. This caused the ship to take on water and eventually sink, leading to the death of many passengers due to a shortage of lifeboats and insufficient rescue attempts.\n\
            stars: 2\n\
            question: What causes seasons on Earth?\n\
            correct answer: Seasons on Earth are caused by the tilt of the Earth's axis and its revolution around the Sun. As the Earth orbits the Sun, the tilt causes different parts of the planet to receive varying amounts of sunlight, resulting in changes in temperature and weather patterns.\n\
            predicted answer: Seasons occur because of the Earth's rotation and its elliptical orbit around the Sun. The tilt of the Earth's axis causes regions to be subjected to different sunlight intensities, which leads to temperature fluctuations and alternating weather conditions.\n\
            stars: 3\n\
            question: How does photosynthesis work?\n\
            correct answer: Photosynthesis is a process by which green plants and some other organisms convert light energy into chemical energy. This occurs as light is absorbed by chlorophyll molecules, and then carbon dioxide and water are converted into glucose and oxygen through a series of reactions.\n\
            predicted answer: In photosynthesis, sunlight is transformed into nutrients by plants and certain microorganisms. Light is captured by chlorophyll molecules, followed by the conversion of carbon dioxide and water into sugar and oxygen through multiple reactions.\n\
            stars: 4\n\
            question: What are the health benefits of regular exercise?\n\
            correct answer: Regular exercise can help maintain a healthy weight, increase muscle and bone strength, and reduce the risk of chronic diseases. It also promotes mental well-being by reducing stress and improving overall mood.\n\
            predicted answer: Routine physical activity can contribute to maintaining ideal body weight, enhancing muscle and bone strength, and preventing chronic illnesses. In addition, it supports mental health by alleviating stress and augmenting general mood.\n\
            stars: 5\n\
            question: {question}\n\
            correct answer: {ground_truth}\n\
            predicted answer: {answer}\n\
            stars:"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        similarity_score = int(response.choices[0].message.content)
        return similarity_score

    # Function to compute Groundedness score
    def calculate_groundedness_score(self, context, answer):
        messages = [
            {"role": "system", "content": "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric."},
            {"role": "user", "content": f"You will be presented with a CONTEXT and an ANSWER about that CONTEXT. You need to decide whether the ANSWER is entailed by the CONTEXT by choosing one of the following rating:\n\
            5: The ANSWER follows logically from the information contained in the CONTEXT.\n\
            4: The ANSWER mostly follows logically from the information contained in the CONTEXT, but with minor inaccuracies.\n\
            3: The ANSWER follows logically from the information contained in the CONTEXT, but still incomplete.\n\
            2: The ANSWER mostly not follows logically from the information contained in the CONTEXT, with major inaccuracies.\n\
            1: The ANSWER is logically false from the information contained in the CONTEXT.\n\
            This rating value should always be an integer between 1 and 5 only, no text. So the rating produced should be 1 or 2 or 3 or 4 or 5.\n\
            Read the passage of information thoroughly and select the correct answer from the five answer labels. Read the CONTEXT thoroughly to ensure you know what the CONTEXT entails.\n\
            \n\
            Note the ANSWER is generated by a computer system, it can contain certain symbols, which should not be a negative factor in the evaluation.\n\
            Independent Examples:\n\
            ## Example Task #1 Input:\n\
            CONTEXT: The Academy Awards, also known as the Oscars are awards for artistic and technical merit for the film industry. They are presented annually by the Academy of Motion Picture Arts and Sciences, in recognition of excellence in cinematic achievements as assessed by the Academy's voting membership. The Academy Awards are regarded by many as the most prestigious, significant awards in the entertainment industry in the United States and worldwide., ANSWER: Oscar is presented every other two years\n\
            ## Example Task #1 Output:\n\
            1\n\
            ## Example Task #2 Input:\n\
            CONTEXT: The Academy Awards, also known as the Oscars are awards for artistic and technical merit for the film industry. They are presented annually by the Academy of Motion Picture Arts and Sciences, in recognition of excellence in cinematic achievements as assessed by the Academy's voting membership. The Academy Awards are regarded by many as the most prestigious, significant awards in the entertainment industry in the United States and worldwide., ANSWER: Oscar is very important awards in the entertainment industry in the United States. And it's also significant worldwide\n\
            ## Example Task #2 Output:\n\
            5\n\
            ## Example Task #3 Input:\n\
            CONTEXT: The Eiffel Tower is one of the most recognizable structures in the world, located in Paris, France. It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair., ANSWER: The Eiffel Tower was built in 1899 and is located in Paris, France.\n\
            ## Example Task #3 Output:\n\
            2\n\
            ## Example Task #4 Input:\n\
            CONTEXT: The Amazon rainforest is the largest tropical rainforest in the world, covering much of northwestern Brazil and extending into Colombia, Peru, and other South American countries. It is known for its biodiversity., ANSWER: The Amazon rainforest is the largest tropical rainforest in the world, located in Brazil, Colombia, Peru, and other South American countries. It is famous for its variety of species, but this answer does not mention that it extends into multiple countries.\n\
            ## Example Task #4 Output:\n\
            3\n\
            ## Example Task #5 Input:\n\
            CONTEXT: Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for its influence on the philosophy of science., ANSWER: Albert Einstein was a physicist who developed the theory of relativity, which is a fundamental concept in modern physics. However, the answer does not mention his influence on the philosophy of science.\n\
            ## Example Task #5 Output:\n\
            4\n\
            \n\
            Reminder: The return values for each task should be correctly formatted as an integer between 1 and 5. Do not repeat the context.\n\
            ## Actual Task Input:\n\
            CONTEXT: {context}, ANSWER: {answer}\n\
            Actual Task Output:"}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        groundedness_score = int(response.choices[0].message.content)
        return groundedness_score

    # Function to compute Relevance score
    def calculate_relevance_score(self, context, question, answer):
        messages = [
            {"role": "system", "content": "You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of an answer in a question-answering task. Your job is to compute an accurate evaluation score using the provided evaluation metric."},
            {"role": "user", "content": f"Relevance measures how well the answer addresses the main aspects of the question, based on the context. Consider whether all and only the important aspects are contained in the answer when evaluating relevance. Given the context and question, score the relevance of the answer between one to five stars using the following rating scale:\n\
            One star: the answer completely lacks relevance\n\
            Two stars: the answer mostly lacks relevance\n\
            Three stars: the answer is partially relevant\n\
            Four stars: the answer is mostly relevant\n\
            Five stars: the answer has perfect relevance\n\
            This rating value should always be an integer between 1 and 5 only, no text. So the rating produced should be 1 or 2 or 3 or 4 or 5.\n\
            \n\
            context: Marie Curie was a Polish-born physicist and chemist who pioneered research on radioactivity and was the first woman to win a Nobel Prize.\n\
            question: What field did Marie Curie excel in?\n\
            answer: Marie Curie was a renowned painter who focused mainly on impressionist styles and techniques.\n\
            stars: 1\n\
            \n\
            context: The Beatles were an English rock band formed in Liverpool in 1960, and they are widely regarded as the most influential music band in history.\n\
            question: Where were The Beatles formed?\n\
            answer: The band The Beatles began their journey in London, England, and they changed the history of music.\n\
            stars: 2\n\
            \n\
            context: The recent Mars rover, Perseverance, was launched in 2020 with the main goal of searching for signs of ancient life on Mars. The rover also carries an experiment called MOXIE, which aims to generate oxygen from the Martian atmosphere.\n\
            question: What are the main goals of Perseverance Mars rover mission?\n\
            answer: The Perseverance Mars rover mission focuses on searching for signs of ancient life on Mars.\n\
            stars: 3\n\
            \n\
            context: The Mediterranean diet is a commonly recommended dietary plan that emphasizes fruits, vegetables, whole grains, legumes, lean proteins, and healthy fats. Studies have shown that it offers numerous health benefits, including a reduced risk of heart disease and improved cognitive health.\n\
            question: What are the main components of the Mediterranean diet?\n\
            answer: The Mediterranean diet primarily consists of fruits, vegetables, whole grains, and legumes.\n\
            stars: 4\n\
            \n\
            context: The Queen's Royal Castle is a well-known tourist attraction in the United Kingdom. It spans over 500 acres and contains extensive gardens and parks. The castle was built in the 15th century and has been home to generations of royalty.\n\
            question: What are the main attractions of the Queen's Royal Castle?\n\
            answer: The main attractions of the Queen's Royal Castle are its expansive 500-acre grounds, extensive gardens, parks, and the historical castle itself, which dates back to the 15th century and has housed generations of royalty.\n\
            stars: 5\n\
            \n\
            context: {context}\n\
            question: {question}\n\
            answer: {answer}\n\
            stars:"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        relevance_score = int(response.choices[0].message.content)
        return relevance_score
    
    # Function to generate llm conclusion
    def generate_llm_conclusion(self, metrics_results):
            # Define the metrics descriptions
            metrics_description = (
                "F1 Score: Measures the closeness in meaning between words in the generated text and the ground truth, value: [0,1].\n"
                "Coherence Score: Evaluates how logically structured and readable the text is, value: [1,5].\n"
                "Similarity Score: Assesses how semantically similar the generated text is to the ground truth, value: [1,5].\n"
                "Groundedness Score: Determines how well the text is supported by the given context, value: [1,5].\n"
                "Relevance Score: Gauges how pertinent the response is to the given question, value: [1,5].\n"
            )

            # Create the prompt with the provided metrics
            prompt = "Based on the following chatbot evaluation metrics, provide a really concise conclusion and interpretation of the chatbot's performance:\n\n"
            prompt += metrics_description + "\n"
            
            for metric, score in metrics_results.items():
                prompt += f"{metric}: {score}\n"
            
            prompt += "\nConclusion and Interpretation:"
            
            # Call the GPT-4 API to generate the conclusion and interpretation
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Extract and return the generated text
            conclusion = response.choices[0].message.content
            return conclusion