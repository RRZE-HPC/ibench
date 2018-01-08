#include <stdio.h>  
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

double (*latency)(int);
int *ninst;

void benchmark(const int N, float freq, char *sofile) {
    struct timeval start, end;
    double benchtime;
    char *instr = strtok(sofile, ".");

    double result;

    // run benchmark
    gettimeofday(&start, NULL);
    result = (*latency)(N);
    gettimeofday(&end, NULL);


    benchtime = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    // divide by 1e6 (usec -> s), ninst (number of instr per loop),
    // N/1e9 (loop count vs. GHz); multiply by frequency
    benchtime = benchtime / (1e6 * *ninst / freq * (N / 1e9));
    printf("%s:%s\t%.3f (clock cycles)\t[DEBUG - result: %f]\n", instr, strlen(instr) + 1 < 8 ? "\t" : "",  benchtime, result);
}
    
int main(int argc, const char *argv[]) {
    // one million runs
    const int N = 1000000;
    float freq = 0.0f;

    // need a target directory containing benchmarks
    if (argc < 2) {
        printf("please specify a directory containing the shared objects with benchmarks to run\n");
        exit(EXIT_FAILURE);
    }

    // did the command line specify a frequency?
    if (argc < 3) {
        printf("Please specify the CPU frequency in GHz. For best results make "
                "sure the frequency is fixed, otherwise SpeedStep/Turbo Boost "
                "might distort the results.\n");
        exit(EXIT_FAILURE);
    }
    freq = atof(argv[2]);
    printf("Using frequency %.2fGHz.\n", freq);

    // perform benchmark for all shared objects in target directory
    DIR *dirp;
    struct dirent *dp;
    struct stat st;
    if ((dirp = opendir(argv[1])) == NULL) {
        perror("opendir");
        exit(EXIT_FAILURE);
    }
    while ((dp = readdir(dirp)) != NULL) {
        // only try .so files
        char *suffix = ".so";
        int lensuffix = strlen(suffix);
        if (strncmp(dp->d_name + strlen(dp->d_name) - lensuffix, ".so", 3))
            continue;

        // load .so
        void *handle;
        size_t len1 = strlen(argv[1]);
        size_t len2 = strlen(dp->d_name);
        // directory might be missing a trailing '/'
        char *relpath;
        if ((relpath = malloc(len1 + len2 + 2)) == NULL) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }
        snprintf(relpath, len1 + len2 + 2, "%s/%s", argv[1], dp->d_name);
        if ((handle = dlopen(relpath, RTLD_LAZY)) == NULL) {
            fprintf(stderr, "dlopen: failed to open %s: %s\n", relpath,
                    dlerror());
            exit(EXIT_FAILURE);
        }
        if ((latency = (double (*)(int))dlsym(handle, "latency")) == NULL) {
            fprintf(stderr, "dlsym: couldn't find function latency in %s: %s\n",
                    relpath, dlerror());
            return (EXIT_FAILURE);
        }
        if ((ninst = (int *)dlsym(handle, "ninst")) == NULL) {
            fprintf(stderr, "dlsym: couldn't find symbol ninst in %s: %s\n",
                    relpath, dlerror());
            return (EXIT_FAILURE);
        }
        free(relpath);

        // do actual benchmark
        benchmark(N, freq, dp->d_name);

        dlclose(handle);
    }

    return 0;
}
