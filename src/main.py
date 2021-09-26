import collections
import enum
import functools
import itertools
import pathlib
import textwrap

import pygad
from PIL import Image, ImageDraw, ImageFont


class Character(enum.Enum):
    # NOTE: could create a-z dynamically with string.ascii_lowercase
    A = 'a'
    B = 'b'
    C = 'c'
    D = 'd'
    E = 'e'
    F = 'f'
    G = 'g'
    H = 'h'
    I = 'i'
    J = 'j'
    K = 'k'
    L = 'l'
    M = 'm'
    N = 'n'
    O = 'o'
    P = 'p'
    Q = 'q'
    R = 'r'
    S = 's'
    T = 't'
    U = 'u'
    V = 'v'
    W = 'w'
    X = 'x'
    Y = 'y'
    Z = 'z'
    # NOTE: should we include punctuation?
    # FULL_STOP = '.'
    # COMMA = ','
    # EXCLAMATION_MARK = '!'
    # QUESTION_MARK = '?'

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


T9_KEYS = {
    2: tuple(Character(x) for x in 'abc'),
    3: tuple(Character(x) for x in 'def'),
    4: tuple(Character(x) for x in 'ghi'),
    5: tuple(Character(x) for x in 'jkl'),
    6: tuple(Character(x) for x in 'mno'),
    7: tuple(Character(x) for x in 'pqrs'),
    8: tuple(Character(x) for x in 'tuv'),
    9: tuple(Character(x) for x in 'wxyz'),
}


class Keypad:
    def __init__(self, keys):
        self.keys = keys
        self.chars = dict(itertools.chain(
            *([(char, key) for char in chars]
              for key, chars in keys.items())
        ))

    def __repr__(self) -> str:
        return repr(self.keys)

    def char_to_key(self, char):
        return int(self.chars[char])

    @staticmethod
    def from_array(array):
        keys = collections.defaultdict(lambda: list())
        for i, c in zip(array, Character):
            keys[i].append(c)
        return Keypad(dict(keys))

    def to_array(self):
        char_keys = []
        for c in Character:
            char_keys.append(self.char_to_key(c))
        return char_keys

    def draw(self):
        img_dim = 1080*3
        out = Image.new("RGB", (img_dim, img_dim), 'lightgray')
        font_large = ImageFont.truetype('Helvetica', 120)
        font_small = ImageFont.truetype('Helvetica', 90)
        d = ImageDraw.Draw(out)

        button_labels = {
            (0, 0): 1,
            (0, 1): 2,
            (0, 2): 3,
            (1, 0): 4,
            (1, 1): 5,
            (1, 2): 6,
            (2, 0): 7,
            (2, 1): 8,
            (2, 2): 9,
            (3, 0): '*',
            (3, 1): 0,
            (3, 2): '#',
        }

        special_keys = {
            0: ['_']
        }

        margin_pc = 0.05
        margin = margin_pc*img_dim
        nrow = 4
        ncol = 3
        button_height = (img_dim-(nrow+1)*margin)/nrow
        button_width = (img_dim-(ncol+1)*margin)/ncol

        for row in range(nrow):
            for col in range(ncol):
                x = margin + col*(button_width+margin)
                y = margin + row*(button_height+margin)

                d.rounded_rectangle(
                    [(x, y),
                    (x+button_width, y+button_height)],
                    outline='gray',
                    fill='white',
                    radius=160,
                    width=6,
                )

                label = button_labels[row, col]
                d.text(
                    (x+button_width/8, y+button_height/8),
                    str(label),
                    font=font_small,
                    fill='gray',
                )

                characters = self.keys.get(label) or special_keys.get(label) or []
                button_label = ', '.join(str(s).upper() for s in characters)
                d.text(
                    (x+button_width/2, y+button_height/2),
                    textwrap.fill(button_label, 10),
                    font=font_large,
                    fill='black',
                    anchor='mm'
                )

        return out


def clean(corpus):
    words = corpus.split()
    without_proper_nouns = ' '.join(w for w in words if w[0].islower())
    valid_chars = {char.value for char in Character} | {' '}
    return ''.join(c for c in without_proper_nouns.lower() if c in valid_chars)


class Typer:
    def __init__(self, keypad: Keypad, dictionary: collections.Counter):
        self.keypad = keypad
        self.dictionary = dictionary

        # Map each word to a key-sequence
        # then add hashes as appropriate
        naive_keypresses = [
            [tuple(keypad.char_to_key(Character(c)) for c in word), word, count]
            for word, count in dictionary.items()
        ]
        naive_keypresses = sorted(naive_keypresses, key=lambda x: (x[0], x[2]))
        textonyms = itertools.groupby(naive_keypresses, key=lambda x: x[0])

        keypresses = {}
        for _, grp in textonyms:
            for i, (naive_keys, word, count) in enumerate(grp):
                # Add disambiguating hash sign for any textonyms
                true_keys = naive_keys + i*('#',)
                keypresses[word] = true_keys

        self.keypresses = keypresses

    def type(self, word):
        return self.keypresses[word]


@functools.lru_cache()
def process_corpus(corpus):
    words = clean(corpus).split()
    dictionary = collections.Counter(words)
    return words, dictionary


def score(keypad, corpus, **kwargs):
    words, dictionary = process_corpus(corpus)
    typer = Typer(keypad, dictionary)

    keystrokes = []
    for word in words:
        keystrokes += typer.type(word)

    return {
        'keystrokes': len(keystrokes),
        'characters': sum(len(w) for w in words),
        'ratio': len(keystrokes) / sum(len(w) for w in words),
        'keypad_array': keypad.to_array(),
        **kwargs
    }


def tn_keypad(n):
    n_char = len(Character)
    key_array = [int(n*i/n_char)+2 for i in range(n_char)]
    return Keypad.from_array(key_array)


if __name__ == '__main__':
    import json
    import tqdm
    import pandas as pd

    DATA_DIR = pathlib.Path(__file__).parents[1]/'data'
    IMG_DIR = pathlib.Path(__file__).parents[1]/'images'

    # First, let's just get the fitness of the regular T9
    with open(DATA_DIR/'train.txt', 'r') as f:
        train = f.read()
    with open(DATA_DIR/'test.txt', 'r') as f:
        test = f.read()

    scores = {}

    # Create some benchmark scores
    t9_keypad = Keypad(keys=T9_KEYS)
    t9_score = score(t9_keypad, test)
    scores['T9'] = t9_score

    for n in range(2, 9):
        keypad = tn_keypad(n)
        scores[f'alphabetical_{n}'] = score(keypad, test)

    # Genetic algorithm options (to become argparse args)
    NUM_GENERATIONS = 350
    NUM_PARENTS_MATING = int(15)
    SOL_PER_POP = int(30)
    MUTATION_NUM_GENES = 1

    def fitness_func(solution, solution_idx):
        keypad = Keypad.from_array(solution)
        # pygad's algo always maximises, so return the -ve ratio
        return -score(keypad, train)['ratio']

    for n in range(2, 9):
        pbar = tqdm.tqdm(total=NUM_GENERATIONS, desc=f'{n} keys')
        solver = pygad.GA(
            # Domain args
            gene_type=int,
            gene_space=[int(i+2) for i in range(n)],
            num_genes=len(Character),
            fitness_func=fitness_func,
            # Configurable args
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            sol_per_pop=SOL_PER_POP,
            mutation_num_genes=MUTATION_NUM_GENES,
            # Callbacks
            on_generation=lambda _: pbar.update(1),
        )
        solver.run()
        pbar.close()

        # solver.plot_fitness()
        solution, solution_fitness, solution_idx = solver.best_solution()

        keypad = Keypad.from_array(solution)
        scores[f'optimised_{n}'] = score(keypad, test, ratio_train=-solution_fitness)

        out = keypad.draw()
        out.save(IMG_DIR/f'optimised_{n}.png')

    with open(DATA_DIR/'results.json', 'w+') as f:
        f.write(json.dumps(scores))
