/** \file
 *  \brief Useful utility functions
 *
 *  More detailed description
 *  Copyright 2015 Christopher Watkins
 */

#include <sys/ioctl.h>
#include <stdio.h>

void progress_bar(int current,
                  int total) {
    struct winsize w;
    ioctl(0, TIOCGWINSZ, &w);

    int hashes_needed = current * (w.ws_col-7) / total + 1;

    if (current == 0) printf("\n");

    if (current * 100 / total < 35)
        printf("\033[31m %3d%%|", current * 100 / total + 1);
    else if (current * 100 / total < 70)
        printf("\033[33m %3d%%|", current * 100 / total + 1);
    else
        printf("\033[32m %3d%%|", current * 100 / total + 1);
    for (int i = 0; i < hashes_needed; ++i) printf("#");
    printf("|\033[m");
    for (int i = 0; i < hashes_needed+7; ++i) printf("\b");
    if (current == total) printf("\n");
    fflush(stdout);

    return;
}
