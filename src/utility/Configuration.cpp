#include "Configuration.hpp"

namespace conf {
    Configuration& get() {
        static Configuration currentConfig;
        return currentConfig;
    }
}