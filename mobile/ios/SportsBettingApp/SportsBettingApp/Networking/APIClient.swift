import Foundation

struct APIClient {
    enum APIError: LocalizedError {
        case invalidResponse
        case serverError(String)
        case decodingError

        var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "We could not reach the server. Please try again."
            case .serverError(let message):
                return message
            case .decodingError:
                return "We received an unexpected response. Please try again later."
            }
        }
    }

    static func post<T: Decodable, U: Encodable>(_ path: String, body: U) async throws -> T {
        var request = URLRequest(url: AppConfig.baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        return try await send(request)
    }

    static func get<T: Decodable>(_ path: String, queryItems: [URLQueryItem] = []) async throws -> T {
        var components = URLComponents(url: AppConfig.baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)
        if !queryItems.isEmpty {
            components?.queryItems = queryItems
        }
        guard let url = components?.url else {
            throw APIError.invalidResponse
        }
        let request = URLRequest(url: url)
        return try await send(request)
    }

    private static func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard 200..<300 ~= httpResponse.statusCode else {
            let message = String(data: data, encoding: .utf8) ?? "Server error."
            throw APIError.serverError(message)
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.decodingError
        }
    }
}
