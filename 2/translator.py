import json


class MachineTranslator:
    def __init__(self):
        # Dictionary-based translation lexicon
        # This is a simplified translation model to demonstrate the concept
        self.eng_to_swahili = {
            # Basic nouns
            "dog": "mbwa",
            "cat": "paka",
            "house": "nyumba",
            "water": "maji",
            "book": "kitabu",

            # Basic verbs
            "eat": "kula",
            "drink": "kunywa",
            "run": "kukimbia",
            "sleep": "kulala",
            "write": "kuandika",

            # Basic adjectives
            "big": "kubwa",
            "small": "ndogo",
            "good": "nzuri",
            "bad": "mbaya",
            "happy": "furaha",

            # Basic phrases
            "hello": "habari",
            "goodbye": "kwaheri",
            "thank you": "asante",
            "how are you": "habari yako"
        }

        # Create reverse dictionary for Swahili to English
        self.swahili_to_eng = {swahili: english for english, swahili in self.eng_to_swahili.items()}

    def translate(self, text, source_lang):
        """
        Translate text between English and Swahili

        Args:
            text (str): Text to translate
            source_lang (str): Source language ('en' or 'sw')

        Returns:
            str: Translated text
        """
        # Convert to lowercase for consistent matching
        text = text.lower()

        # Select appropriate dictionary based on source language
        if source_lang == 'en':
            translation_dict = self.eng_to_swahili
        elif source_lang == 'sw':
            translation_dict = self.swahili_to_eng
        else:
            return "Unsupported language"

        # Split text into words
        words = text.split()

        # Translate each word
        translated_words = []
        for word in words:
            # Look up translation, use original word if not found
            translated_word = translation_dict.get(word, word)
            translated_words.append(translated_word)

        # Join translated words
        return " ".join(translated_words)


# Create translator instance
translator = MachineTranslator()


# Test translation examples
def test_translations():
    print("--- English to Swahili Translation Tests ---")
    eng_tests = [
        "hello",
        "big dog",
        "drink water",
        "how are you"
    ]

    for test in eng_tests:
        print(f"English: {test}")
        print(f"Swahili: {translator.translate(test, 'en')}")
        print()

    print("--- Swahili to English Translation Tests ---")
    swahili_tests = [
        "habari",
        "mbwa kubwa",
        "kunywa maji",
        "habari yako"
    ]

    for test in swahili_tests:
        print(f"Swahili: {test}")
        print(f"English: {translator.translate(test, 'sw')}")
        print()


# Run translation tests
test_translations()
