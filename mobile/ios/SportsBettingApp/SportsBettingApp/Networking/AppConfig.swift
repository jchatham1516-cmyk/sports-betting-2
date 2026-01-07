import Foundation

enum AppConfig {
    /// BASE_URL is defined in Info.plist.
    /// - Simulator default: http://127.0.0.1:8000
    /// - Physical iPhone: set BASE_URL to your Mac's LAN IP (e.g. http://192.168.1.10:8000)
    static var baseURL: URL {
        guard let baseURLString = Bundle.main.object(forInfoDictionaryKey: "BASE_URL") as? String,
              let url = URL(string: baseURLString) else {
            return URL(string: "http://127.0.0.1:8000")!
        }
        return url
    }

    static var appVersion: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        return "v\(version ?? "1.0") (\(build ?? "1"))"
    }
}
