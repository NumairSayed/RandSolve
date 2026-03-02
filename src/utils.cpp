/**
 * @file utils.cpp
 * @brief Utility implementations for the RandNLA project.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

namespace randnla::utils {

void print_separator(const std::string& title, int width) {
    std::string line(width, '=');
    std::cout << "\n" << line << "\n";
    if (!title.empty()) {
        int pad = (width - static_cast<int>(title.size()) - 2) / 2;
        std::cout << std::string(std::max(0, pad), ' ')
                  << " " << title << " "
                  << std::string(std::max(0, pad), ' ') << "\n";
        std::cout << line << "\n";
    }
}

void print_stats(const std::string& label,
                 double residual,
                 double elapsed_ms,
                 double speedup) {
    std::cout << std::left  << std::setw(30) << label
              << std::right << std::setw(14) << std::scientific
                            << std::setprecision(4) << residual
              << std::setw(14) << std::fixed << std::setprecision(2)
                            << elapsed_ms << " ms"
              << std::setw(10) << std::setprecision(2) << speedup << "x"
              << "\n";
}

} // namespace randnla::utils