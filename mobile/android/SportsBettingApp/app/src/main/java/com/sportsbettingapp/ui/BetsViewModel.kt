package com.sportsbettingapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.Bet
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class BetsViewModel(
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _betsState = MutableStateFlow<UiState<List<Bet>>>(UiState.Idle)
    val betsState: StateFlow<UiState<List<Bet>>> = _betsState

    fun loadBets(date: String, sport: String) {
        _betsState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val bets = repository.getBets(date, sport)
                _betsState.value = UiState.Success(bets)
            } catch (exception: Exception) {
                _betsState.value = UiState.Error(exception.localizedMessage ?: "Failed to load bets")
            }
        }
    }
}
