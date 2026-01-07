import Foundation

enum Sport: String, CaseIterable, Identifiable, Codable {
    case nba
    case nfl
    case nhl

    var id: String { rawValue }
}

struct RunCreateRequest: Codable {
    let sport: String
    let gameDate: String
    let settings: [String: String]

    enum CodingKeys: String, CodingKey {
        case sport
        case gameDate = "game_date"
        case settings
    }
}

struct RunCreateResponse: Codable, Identifiable {
    let runId: String
    var id: String { runId }

    enum CodingKeys: String, CodingKey {
        case id
        case runId = "run_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let id = try container.decodeIfPresent(String.self, forKey: .id) {
            runId = id
        } else if let runId = try container.decodeIfPresent(String.self, forKey: .runId) {
            self.runId = runId
        } else if let idValue = try container.decodeIfPresent(Int.self, forKey: .id) {
            runId = String(idValue)
        } else if let runValue = try container.decodeIfPresent(Int.self, forKey: .runId) {
            runId = String(runValue)
        } else {
            runId = ""
        }
    }
}

struct PredictionRow: Codable, Identifiable {
    let id = UUID()
    let homeTeam: String?
    let awayTeam: String?
    let primaryRecommendation: String?
    let confidence: String?
    let confidenceTier: String?
    let valueTier: String?
    let price: String?

    enum CodingKeys: String, CodingKey {
        case homeTeam = "home"
        case awayTeam = "away"
        case primaryRecommendation = "primary_recommendation"
        case confidence
        case confidenceTier = "confidence_tier"
        case valueTier = "value_tier"
        case price
    }
}

struct TrackedBetRow: Codable, Identifiable {
    let id = UUID()
    let matchup: String?
    let pick: String?
    let market: String?
    let units: Double?
    let result: String?
}
