#ifndef CSV_HPP
#define CSV_HPP

#include <algorithm>
#include <fstream>
#include <vector>
#include <exception>
#include <cstdint>
#include <string>
#include <mutex>
#include <sstream>
#include <iostream>

namespace CSV
{
    typedef uint64_t n;

    struct color { uint8_t r = 0; uint8_t g = 0; uint8_t b = 150; };
    color red = { 205, 0, 150 };
    color green = { 0, 255, 150 };

    std::ostringstream log_buff("");
    std::mutex log_buff_mutex;
    void log(std::string str, color rgb, bool display)
    {
        {
            std::lock_guard<std::mutex> guard(log_buff_mutex);
            log_buff << "\033[38;2;" + std::to_string(rgb.r) + ";" + std::to_string(rgb.g) + ";" + std::to_string(rgb.b) + "m" + str + "\033[0m\n";
        }
        if (display)
        {
            std::cerr << log_buff.str();
            log_buff.str("");
            log_buff.clear();
        }
    }

    template <typename T>
    std::vector<std::vector<T>> loadCSVn(std::string path)
    {
        std::ifstream r(path);
        if (!r.is_open())
        {
            log("[CSV::loadCSVn]: Failed to open " + path + ". Will return an empty vector.", red, true);
            return std::vector<std::vector<T>>(0);
        }

        n elems = 0;
        {
            std::string headers("");
            std::getline(r, headers);
            elems = std::count(headers.begin(), headers.end(), ',');

            if (elems == 0)
            {
                log("[CSV::loadCSVn]: File " + path + " contains invalid headers. Will return an empty vector.", red, true);
                return std::vector<std::vector<T>>(0);
            }
            else
            {
                ++elems;
                log("[CSV::loadCSVn]: Headers red from " + path + ":\n" + headers, green, false);
            }
        }

        std::vector<std::vector<T>> column;
        std::string buffer("");
        std::string elemstr("");

        try
        {
            while (std::getline(r, buffer))
            {
                std::vector<T> row(elems);
                std::stringstream strstr(buffer);

                for (n i = 0; std::getline(strstr, elemstr, ','); ++i)
                    row[i] = std::stof(elemstr);

                column.push_back(row);
            }
        }
        catch (const std::exception& e) { log("[CSV::loadCSVn]: Failed to parse CSV:\n" + std::string(e.what()), red, true); }

        log("[CSV::loadCSVn]: Finished reading CSV file.", { 0, 255, 150 }, true);

        return column;
    }

    std::vector<std::vector<std::string>> loadCSVstr(std::string path)
    {
        std::ifstream r(path);
        if (!r.is_open())
        {
            log("[CSV::loadCSVstr]: Failed to open " + path + ". Will return an empty vector.", red, true);
            return std::vector<std::vector<std::string>>();
        }

        n elems = 0;
        {
            std::string headers("");
            std::getline(r, headers);
            elems = std::count(headers.begin(), headers.end(), ',');

            if (elems == 0)
            {
                log("[CSV::loadCSVstr]: File " + path + " contains invalid headers. Will return an empty vector.", red, true);
                return std::vector<std::vector<std::string>>(0);
            }
            else
            {
                ++elems;
                log("[CSV::loadCSVstr]: Headers read from " + path + ":\n" + headers, green, false);
            }
        }

        std::vector<std::vector<std::string>> column;
        std::string buffer("");

        try
        {
            while (std::getline(r, buffer))
            {
                std::vector<std::string> row;
                std::stringstream strstr(buffer);
                std::string ce; // current elem
                char c;
                bool iq = false; // in quotes
    
                while (strstr.get(c))
                {
                    if (c == '\\') 
                    {
                        if (strstr.peek() == '"' || strstr.peek() == '\\' || strstr.peek() == ',') ce += strstr.get();
                        else ce += c;
                    }
                    else if (c == '"')
                    {
                        if (iq && strstr.peek() == '"')
                        {
                            ce += '"';
                            strstr.get(c);
                        }
                        else iq = !iq;
                    }
                    else if (c == ',' && !iq)
                    {
                        row.push_back(ce);
                        ce.clear();
                    }
                    else ce += c;
                }

                if (!ce.empty() || buffer.back() == ',') row.push_back(ce);
                column.push_back(row);
            }
        }
        catch (const std::exception& e)
        {
            log("[CSV::loadCSVstr]: Failed to parse CSV:\n" + std::string(e.what()), red, true);
        }

        log("[CSV::loadCSVstr]: Finished reading CSV file.", green, true);

        return column;
    }

    template <typename T>
    bool saveCSVn(std::vector<std::vector<T>> csv, std::string header, std::string path)
    {
        std::ofstream w(path);
        if (!w.is_open()) return false;
        w << header;
        if (!bool(csv.size())) return true;

        n elems = csv[0].size();
        for (std::vector<T> row : csv)
        {
            w << '\n' << row[0];
            for (n elem = 1; elem < elems; ++elem) w << ',' << row[elem];
        }

        return true;
    }

    bool saveCSVstr(std::vector<std::vector<std::string>> csv, std::string header, std::string path)
    {
        std::ofstream w(path);
        if (!w.is_open()) return false;
        w << header;
        if (!bool(csv.size())) return true;

        n elems = csv[0].size();
        for (std::vector<std::string> row : csv)
        {
            w << "\n\"" << row[0] << '"';
            for (n elem = 1; elem < elems; ++elem) w << ",\"" << row[elem] << '"';
        }

        return true;
    }
};
#endif
