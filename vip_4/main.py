from solution import Assignment
import sys
import os


def main():
    if len(sys.argv) > 3:
        print('Too many parameters!')
        exit(1)

    if 'tfidf' in sys.argv and 'common' in sys.argv:
        print('Use either "common" or "tfidf", not both!')
        exit(1)

    if 'tfidf' not in sys.argv and 'common' not in sys.argv and 'clean' not in sys.argv and len(sys.argv) > 1:
        print('\tIncorrect values passed! Allowed values: [tfidf, common, clean]')
        print('\t"clean" may be passed with "tfidf" or "common"')
        print('\t"tfidf" and "common" cannot be passed together!')
        print('\tDefault retrieval method: [tfidf] (tfidf w/ cosine similarity)')
        exit(1)

    if 'clean' in sys.argv:
        print('Option "clean" passed, deleting any previously generated bags of words...', end='', flush=True)

        try:
            os.remove(Assignment.save_filename)
        except OSError:
            pass

        print('DONE')

    # Beginning of assignment solution
    solution = Assignment('./dataset/')

    if not solution.loaded:
        solution.codebook_generation(k_clusters=500)
        solution.indexing()

    solution.retrieving(strat='common' if 'common' in sys.argv else 'tfidf')


if __name__ == '__main__':
    main()
