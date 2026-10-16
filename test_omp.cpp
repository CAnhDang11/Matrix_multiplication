#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp critical
        std::cout << "Hello from thread "
                  << omp_get_thread_num() << '\n';
    }

    std::cout << "Number of threads: "
              << omp_get_max_threads() << '\n';
    return 0;
}

