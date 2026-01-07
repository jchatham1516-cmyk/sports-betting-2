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
    let runId: Int
    var id: Int { runId }

    enum CodingKeys: String, CodingKey {
        case id
        case runId = "run_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let id = try container.decodeIfPresent(Int.self, forKey: .id) {
            runId = id
        } else if let runId = try container.decodeIfPresent(Int.self, forKey: .runId) {
            self.runId = runId
        } else {
            runId = 0
        }
    }
}

struct PredictionRow: Codable, Identifiable {
    let id = UUID()
    let homeTeam: String?
    let awayTeam: String?
    let primaryRecommendation: String?
    let confidenceTier: String?
    let valueTier: String?
    let price: String?

    enum CodingKeys: String, CodingKey {
        case homeTeam = "home"
        case awayTeam = "away"
        case primaryRecommendation = "primary_recommendation"
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
