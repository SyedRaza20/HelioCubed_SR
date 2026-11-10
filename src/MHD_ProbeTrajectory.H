#pragma once

#include "MHD_Time.H"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace HelioCubed
{
struct ProbeTrajectory
{
  std::vector<double> time, radius, latitude, longitude;

  ProbeTrajectory() = default;

  explicit ProbeTrajectory(const std::string& filename)
  {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Unable to open probe trajectory: " + filename);
    std::string line;
    while (std::getline(file, line))
    {
      if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
      double yearValue, dayValue, r, lat, lon;
      std::istringstream row(line);
      if (!(row >> yearValue >> dayValue >> r >> lat >> lon))
        throw std::runtime_error("Invalid probe trajectory row in: " + filename);
      // Retain the existing whole-day interpretation of trajectory DOY values.
      const int year = static_cast<int>(yearValue);
      const int day = static_cast<int>(dayValue);
      const double epoch = year + (day - 1)/(gregorianLeapYear(year) ? 366.0 : 365.0);
      if (!time.empty() && epoch <= time.back())
        throw std::runtime_error("Probe trajectory times must increase: " + filename);
      time.push_back(epoch);
      radius.push_back(r);
      latitude.push_back(lat);
      longitude.push_back(lon);
    }
    if (time.size() < 2)
      throw std::runtime_error("Probe trajectory needs at least two rows: " + filename);
  }
};

inline const ProbeTrajectory& probeTrajectory(const std::string& filename)
{
  // Trajectories are run inputs; load each distinct file once on each rank.
  static std::map<std::string, ProbeTrajectory> cache;
  auto entry = cache.find(filename);
  if (entry == cache.end())
    entry = cache.emplace(filename, ProbeTrajectory(filename)).first;
  return entry->second;
}

}
