#include <type_traits>
#include <string>
#include <sstream>
#include <concepts>
#include <iostream>

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <iterator>

template <typename Stringable>
auto inline to_string(Stringable v) -> decltype(std::to_string(v))
{
    return std::to_string(v);
}

template <typename T>
concept has_to_string = requires(T const &x) {
    {
        to_string(x)
    } -> std::convertible_to<std::string>;
};

template <typename T>
concept has_forward_iterator = requires(T v) {
    {
        v.begin()
    } -> std::forward_iterator;
};

template <has_forward_iterator T>
inline std::ostream &iter_to_string(std::ostream &stream, T const &container, std::pair<std::string, std::string> wrapper)
{
    stream << wrapper.first;
    auto iter = container.begin();
    if (iter != container.end())
    {
        stream << *iter;
    }
    iter++;
    for (; iter != container.end(); iter++)
    {
        stream << ", " << *iter;
    }
    stream << wrapper.second;
    return stream;
}

template <has_forward_iterator T>
inline std::ostream &operator<<(std::ostream &stream, T const &container)
{
    return iter_to_string(stream, container, std::make_pair("[", "]"));
}

template <has_to_string T>
inline std::ostream &operator<<(std::ostream &stream, std::set<T> const &container)
{
    return iter_to_string(stream, container, std::make_pair("{", "}"));
}

template <has_to_string T1, has_to_string T2>
inline std::ostream &operator<<(std::ostream &stream, std::pair<T1, T2> const &x)
{
    stream << x.first << ": " << x.second;

    return stream;
}

int main()
{
    std::cout << "Hello World\n";
    std::cout << 10 << "\n";
    std::cout << std::vector{1, 2, 3} << "\n";
    std::cout << std::set{1, 2, 3} << "\n";
    std::cout << std::make_pair(1, 2) << "\n";
    std::cout << std::map<int, int>{{1, 2}, {3, 4}} << "\n";

    return 0;
}
