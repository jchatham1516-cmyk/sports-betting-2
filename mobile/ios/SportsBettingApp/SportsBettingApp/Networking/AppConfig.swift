import Foundation

enum AppConfig {
    static var baseURL: URL {
        guard let baseURLString = Bundle.main.object(forInfoDictionaryKey: "BASE_URL") as? String,
              let url = URL(string: baseURLString) else {
            return URL(string: "http://localhost:8000")!
        }
        return url
    }

    static var appVersion: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        return "v\(version ?? "1.0") (\(build ?? "1"))"
    }
}
