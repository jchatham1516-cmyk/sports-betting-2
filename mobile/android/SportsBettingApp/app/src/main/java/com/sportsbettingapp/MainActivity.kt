package com.sportsbettingapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.sportsbettingapp.ui.BetsViewModel
import com.sportsbettingapp.ui.PredictionsViewModel
import com.sportsbettingapp.ui.RunViewModel
import com.sportsbettingapp.ui.screens.BetsScreen
import com.sportsbettingapp.ui.screens.DisclaimerScreen
import com.sportsbettingapp.ui.screens.PredictionsScreen
import com.sportsbettingapp.ui.screens.RunScreen
import com.sportsbettingapp.ui.screens.SettingsScreen
import com.sportsbettingapp.ui.theme.SportsBettingTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SportsBettingTheme {
                SportsBettingApp()
            }
        }
    }
}

private sealed class Screen(val route: String, val label: String) {
    data object Run : Screen("run", "Run")
    data object Bets : Screen("bets", "Bets")
    data object Settings : Screen("settings", "Settings")
    data object Predictions : Screen("predictions", "Predictions")
    data object Disclaimer : Screen("disclaimer", "Disclaimer")
}

@Composable
private fun SportsBettingApp() {
    val navController = rememberNavController()
    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = currentBackStackEntry?.destination?.route
    val showBottomBar = currentRoute in listOf(Screen.Run.route, Screen.Bets.route, Screen.Settings.route)

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                BottomNavBar(navController = navController)
            }
        }
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Run.route
        ) {
            composable(Screen.Run.route) {
                val viewModel: RunViewModel = viewModel()
                RunScreen(
                    viewModel = viewModel,
                    onRunSuccess = { runId ->
                        navController.navigate("${Screen.Predictions.route}/$runId")
                    },
                    contentPadding = padding
                )
            }
            composable(Screen.Bets.route) {
                val viewModel: BetsViewModel = viewModel()
                BetsScreen(viewModel = viewModel, contentPadding = padding)
            }
            composable(Screen.Settings.route) {
                SettingsScreen(
                    onDisclaimerClick = { navController.navigate(Screen.Disclaimer.route) },
                    contentPadding = padding
                )
            }
            composable("${Screen.Predictions.route}/{runId}") { backStackEntry ->
                val runId = backStackEntry.arguments?.getString("runId") ?: ""
                val viewModel: PredictionsViewModel = viewModel()
                PredictionsScreen(runId = runId, viewModel = viewModel, contentPadding = padding)
            }
            composable(Screen.Disclaimer.route) {
                DisclaimerScreen(contentPadding = padding)
            }
        }
    }
}

@Composable
private fun BottomNavBar(navController: NavHostController) {
    val items = listOf(Screen.Run, Screen.Bets, Screen.Settings)
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    androidx.compose.material3.NavigationBar {
        items.forEach { screen ->
            androidx.compose.material3.NavigationBarItem(
                selected = currentRoute == screen.route,
                onClick = {
                    navController.navigate(screen.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                label = { Text(screen.label) },
                icon = {}
            )
        }
    }
}
