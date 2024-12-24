#include "ANNA/Data-Prep/CSV.hpp"

using namespace CSV;

int main()
{
    std::string header = "ID,Age";
    std::vector<std::vector<float>> data =
    {
        { 0, 18 },
        { 1, 20 },
        { 2, 22 }
    };

    if (saveCSV(data, header, "data.csv")) log("Success.", green, true);
    else log("Fail.", red, true);

    return 0;
}