import sys
from PyQt5 import QtCore
from solution import AssignmentSection


def main():
    if len(sys.argv) > 2:
        print('Only one dataset name is allowed!')
        return

    if len(sys.argv) < 2:
        AssignmentSection.beethoven()       # Beethoven Dataset
        AssignmentSection.mate_vase()       # mat_vase Dataset
        AssignmentSection.shiny_vase()      # shiny_vase Dataset
        AssignmentSection.shiny_vase_two()  # shiny_vase2 Dataset
        AssignmentSection.buddha()          # Buddha Dataset
        AssignmentSection.face()            # face Dataset
        return

    dataset_name = sys.argv[1]

    if dataset_name == 'beethoven':
        AssignmentSection.beethoven()
    elif dataset_name == 'mat_vase':
        AssignmentSection.mate_vase()
    elif dataset_name == 'shiny_vase':
        AssignmentSection.shiny_vase()
    elif dataset_name == 'shiny_vase2':
        AssignmentSection.shiny_vase_two()
    elif dataset_name == 'buddha':
        AssignmentSection.buddha()
    elif dataset_name == 'face':
        AssignmentSection.face()
    else:
        print('Invalid dataset name passed! Available dataset names: [beethoven, mat_vase, shiny_vase, shiny_vase2, buddha, ' +
              'face].\nDon\'t pass a name to display results on all of them!')


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)  # Fixes qt warning about being opened from a plugin
    main()

